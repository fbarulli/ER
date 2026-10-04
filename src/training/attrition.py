"""Validated, frozen accounting from pair supply to the training objective."""
from collections import Counter
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, model_validator


class StageCensus(BaseModel):
    rows: int = Field(ge=0)
    unique_pairs: int = Field(ge=0)
    populations: dict[str, int]

    @model_validator(mode='after')
    def complete(self):
        if sum(self.populations.values()) != self.rows or self.unique_pairs > self.rows:
            raise ValueError('stage census does not account for every row')
        return self


class PairAttrition(BaseModel):
    label: Literal['positive', 'negative']
    stages: dict[Literal['stored', 'post_holdout', 'post_selection', 'post_downsample'], StageCensus]
    transitions: dict[str, dict[str, int]]
    objective_presentations: int = Field(ge=0)
    objective_unique_pairs: int = Field(ge=0)
    surviving_supply_unique_pairs: int = Field(ge=0)
    excluded_no_objective_counterpart: int = Field(ge=0)
    objective_derived_unique_pairs: int = Field(ge=0)

    @model_validator(mode='after')
    def complete(self):
        if set(self.stages) != {'stored','post_holdout','post_selection','post_downsample'}:
            raise ValueError('missing attrition stage')
        final = self.stages['post_downsample'].unique_pairs
        if final != self.surviving_supply_unique_pairs + self.excluded_no_objective_counterpart:
            raise ValueError('objective attrition does not reconcile')
        if self.objective_unique_pairs != self.surviving_supply_unique_pairs + self.objective_derived_unique_pairs:
            raise ValueError('derived objective pairs do not reconcile')
        names=list(self.stages)
        for before,after in zip(names,names[1:]):
            delta=self.transitions[f'{before}->{after}']
            if self.stages[before].rows-delta['removed_rows']+delta['added_rows'] != self.stages[after].rows:
                raise ValueError('stage transition does not reconcile')
        return self


class AttritionLedger(BaseModel):
    version: Literal[1] = 1
    positive: PairAttrition
    negative: PairAttrition
    objective_populations: dict[str, int]
    objective_rows: int = Field(ge=0)

    @model_validator(mode='after')
    def complete(self):
        if sum(self.objective_populations.values()) != self.objective_rows:
            raise ValueError('objective populations do not reconcile')
        return self


def build_attrition_ledger(bundle, fold):
    def edges(rows):
        return Counter(tuple(sorted(map(int,row))) for row in np.asarray(rows).reshape(-1,2))
    audit={int(row['copy_payload_idx']): row for row in [*bundle.get('mask_audit',[]),*bundle.get('hard_negative_mask_audit',[])]}
    sources={tuple(sorted(map(int,pair))):str(source) for pair,source in zip(bundle['neg'],bundle['neg_sources'],strict=True)}
    def population(pair,label):
        entries=[audit[i] for i in pair if i in audit]
        if entries:
            return 'augmented:'+str(entries[0].get('generation_variant') or entries[0].get('target_mode','unknown'))
        return 'original' if label=='positive' else sources.get(pair,'additional_negative')
    def census(rows,label):
        counts=edges(rows)
        pops=Counter()
        for pair,count in counts.items(): pops[population(pair,label)]+=count
        return StageCensus(rows=sum(counts.values()),unique_pairs=len(counts),populations=dict(pops))
    triples=fold['objective'].get('triples',[])
    dataset=fold['objective']['dataset']
    if not triples:
        return None  # Pair objectives already consume their selected pair rows directly.
    train=set(map(str,fold['tr_bc']))
    def ledger(label,stored,selected,downsampled,objective):
        held=[row for row in stored if all(str(bundle['row_bc'][int(i)]) in train for i in row)]
        rows={'stored':stored,'post_holdout':held,'post_selection':selected,'post_downsample':downsampled}
        stages={name:census(value,label) for name,value in rows.items()}
        transitions={}
        names=list(rows)
        for a,b in zip(names,names[1:]):
            before,after=edges(rows[a]),edges(rows[b])
            transitions[f'{a}->{b}']={'removed_rows':sum((before-after).values()),'added_rows':sum((after-before).values())}
        supplied=set(edges(downsampled)); consumed=set(edges(objective)); surviving=len(supplied&consumed)
        return PairAttrition(label=label,stages=stages,transitions=transitions,
            objective_presentations=len(objective),objective_unique_pairs=len(consumed),
            surviving_supply_unique_pairs=surviving,excluded_no_objective_counterpart=len(supplied)-surviving,
            objective_derived_unique_pairs=len(consumed-supplied))
    return AttritionLedger(
        positive=ledger('positive',bundle['pos'],fold['train_pos'],fold['train_all'],[(a,b) for a,b,c in triples]),
        negative=ledger('negative',bundle['neg'],fold['tr_negs'],fold['tr_negs'],[(a,c) for a,b,c in triples]),
        objective_populations=dict(Counter(dataset['population'])),objective_rows=len(triples)).model_dump(mode='json')
