# What the model sees — training pairs (masking.md)

Generated from the OFFICIAL chain: `clean_sku_text` (number-token reference strip included)
→ `DATA_PIPE.pairs.build_training_data` → masking augmentation (extent varies 5–15%).

**Pair counts** — positives: 26,767 · masked positives: +4,015 · hard negatives: 28,465 · total training pairs: 59,247

Mask token: `` ` `` — the anchor is noised, the pair target (canonical) stays clean.

## Positive pairs — (clean sku, canonical of SAME gtin)

| # | anchor — clean sku | positive — canonical (own gtin) |
|---|---|---|
| 1 | relentless origin energy drink volume caffeine energy source vitamins guarana health claims good source vitamins energy boosting material type metal c | relentless energy origin_energy_drink origin_energy origin_energy_drink_energy erythritol_sucralose_sucrose_volume sucralose_sucrose_volume_carbonizat |
| 2 | julmust nyg rda material type flexible type bag volume carbonization carbonated juice content sweetener sugar naturally derived natural | nyg rda soda gently_carbonated_type_bag type_bag carbonated_type_bag material_type_flexible type_flexible carbonated |
| 3 | hildon natural mineral water delightfully still type caffeine contains minerals calcium magnesium iron juice content energy source vitamins water type | hildon water natural_mineral_water_delightfully mineral_water_delightfully_still water_delightfully_still_type delightfully_still_type_caffeine iron_j |
| 4 | energizing black tea mix peach concentrate format powder electrolytes naturally derived natural water type coconut material type flexible juice conten | cure hydration peach tea energizing_black_tea_mix black_tea_mix_peach energizing_black_tea black_tea_mix tea_mix_peach added_sugar |
| 5 | mc cafe frappe mocha iced coffee caffeine sweetener sugar coffee type mocha carbonization still flavour mocha coffee type rtd coffee style mocha juice | mccaf coffee mocha mc_cafe_frappe_mocha cafe_frappe_mocha frappe_mocha mocha_coffee still |
| 6 | grape juice red ecosana material type glass type flavour grape volume juice content made grape juice features concentrate caffeine health claims added | ecosana juice grape_juice_red grape_juice juice_red grape_artificial_ingredients_artificial grape added_sugar still |
| 7 | vita organic kombucha organic classic flavor sustainable sourcing organic naturally derived natural type volume caffeine material type glass carboniza | vitaorganic organic_kombucha_organic_classic kombucha_organic_classic_flavor organic_classic_flavor_sustainable classic_flavor_sustainable_sourcing ko |
| 8 | peace tea just peachy carbonization still juice content naturally derived natural flavours flavour tea volume artificial ingredients artificial colour | peace tea peach tea georgia peace_tea_georgia_peach cane_sugar_sucralose_fructose peace_tea_georgia tea_georgia_peach still |
| 9 | light feeling kombucha blackcurrant klp type material type glass juice content volume sustainable sourcing organic health claims probiotic carbonizati | kevytolo light_feeling_kombucha_blackcurrant feeling_kombucha_blackcurrant_klp feeling_kombucha_blackcurrant kombucha_blackcurrant_klp kombucha_blackc |
| 10 | towne club vanilla cream soda volume flavour vanilla type juice content | towne club vanilla soda towne_club_vanilla_cream club_vanilla_cream_soda soda_volume_473_flavour volume_473_flavour_vanilla 473_flavour_vanilla_type |

## Hard-negative pairs — (clean sku, canonical of DIFFERENT gtin, gate hard_no)

| # | anchor — clean sku | negative — canonical (other gtin) |
|---|---|---|
| 1 | mineral water bezoya contains minerals calcium magnesium volume juice content water type mineral material type paper carbonization still type | bezoya water mineralization_very_weak_carafe very_weak_carafe_volume weak_carafe_volume_contains carafe_volume_contains_minerals water_bezoya_natural_ |
| 2 | sonatural fresh juice cucumber celery flavour cucumber volume type carbonization still made cucumber | so natural raspberry juice pomegranate_raspberry sonatural_fruit_juice_fresh fruit_juice_fresh_pomegranate juice_fresh_pomegranate_raspberry fresh_pom |
| 3 | natural mineral water hepar energy source vitamins juice content contains minerals calcium magnesium naturally derived natural material type paper wat | h par water score_carbonization_still_mineral mineral_nutri_score_carbonization water_plate_natural_naturally plate_natural_naturally_derived content_ |
| 4 | evolution orange juice fl. oz. volume diets kosher material type plastic flavour orange health claims antioxidant added sugar good source vitamins jui | evolution fresh juice defense_up fruit_juice_smoothie_sustainable juice_smoothie_sustainable_sourcing organic_volume_946_sports volume_946_sports_ingr |
| 5 | bar le duc natural mineral water business suit naturally derived natural carbonization carbonated juice content type liquid health claims low salt vol | bar le duc water carbonated_caffeine_15_juice carbonated_sparkling_volume_material 15_juice_content_carbonization mineral_water_carbonated_caffeine wa |
| 6 | sonatural fresh juice pomegranate vitality shot material type flexible volume made pomegranate carbonization still flavour pomegranate | so natural lemon juice lemon_mint juice_cooled_squeezed_lemon cooled_squeezed_lemon_mint squeezed_lemon_mint_sonatural lemon_mint_sonatural_bottle. st |
| 7 | conad natural mineral water sparkling volume juice content carbonization sparkling water type mineral sustainable packaging recycled materials type po | conad water natural_mineral_water_slightly mineral_water_slightly_sparkling mineral_water_slightly water_slightly_sparkling water_slightly carbonated  |
| 8 | evolution fresh organic cold pressed essential greens vegetable fruit juice blend fl. oz. naturally derived natural flavours natural juice content vol | evolution fresh orange juice volume_450_made_orange 450_made_orange cold_pressed_orange_juice cold_pressed_orange volume_450_made still |
| 9 | cock bull diet ginger beer soda ideal mixer cocktails mocktails bartenders premium quality perfect mixed drinks refreshing fla vor profile type free c | cock n bull ginger soda diet_ginger_beer_soda states_health_claims_sugar cock_bull_diet_ginger bull_diet_ginger_beer derived_natural_type_geographic c |
| 10 | zagori go green mineral water water type mineral volume caffeine sustainable packaging recycled materials carbonization still juice content contains m | zagori water zagori_natural_mineral_water zagori_natural_mineral mineral_material_type_glass carbonization_still_zagori_natural still_zagori_natural_m |

## Masked positives — anchor noised at a random 5–15% extent

| # | anchor — masked clean sku (extent) | positive — canonical (own gtin) |
|---|---|---|
| 1 | zingo tropical pet ` type glass immune support ingredients vitamin ` ` carbonated caffeine type sweetener sugar volume juice ` naturally derived natur (15%) | zingo zingo_tropical natural_fragrances_natural_flavours material_type_glass_immune type_glass_immune_support fragrances_natural_flavours carbonated |
| 2 | sonatural shot drink fresh fruit apple coconut water lemon ` carbon material type glass type volume made lemon coconut apple carbonization still flavo (7%) | so natural apple coconut water lemon_coconut_apple lemon_coconut coconut_apple sonatural_shot_drink_fresh shot_drink_fresh_fruit still |
| 3 | tea zone popping ` popping boba black ` flavorful beverage sensation jar peach material type glass carbonization still contains minerals ` flavour pea (10%) | tea zone peach tea pearls popping tea_zone_popping_pearls zone_popping_pearls_popping popping_pearls_popping_boba still |
| 4 | pear juice unclarified rembowscy type naturally derived ` juice features smoothie sweetener sugar volume flavour pear caffeine carbonization still jui (8%) | rembowscy juice pear_juice_unclarified_rembowscy juice_unclarified_rembowscy_type unclarified_rembowscy_type_naturally rembowscy_type_naturally_derive |
| 5 | concentrate cold ` coffee perfect instant iced coffee cold ` coffee hot coffee original artificial ` gmo coffee type arabica mocha free caffeine volum (8%) | javy coffee hot_coffee_original_artificial coffee_original_artificial_ingredients type_arabica_mocha_free arabica_mocha_free_caffeine caffeine_volume_ |
| 6 | sam cola diet soda free allergens diets kosher ` juice content health claims low calories calories (6%) | sam s cola cola soda cola_diet_soda diet_soda cola_diet cola_diet_soda_free diet_soda_free_allergens carbonated diet |
| 7 | nature goodness aloe vera drink pulp pineapple flavor refreshing beverage real aloe vera juice sweetener cane sugar ` water type aloe vera naturally d (8%) | nature s goodness aloe juice aloe_vera vera_drink_pulp_pineapple drink_pulp_pineapple_flavor drink_pulp_pineapple pulp_pineapple_flavor pulp still |
| 8 | tonic spruce shoots organic eco brewery volume juice ` sustainable sourcing organic ` carbonated type naturally derived natural ` material type glass  (11%) | ekobryggeriet tonic water spruce_shoots spruce shoots spruce_shoots_tonic_water shoots_tonic_water_drinkmix carbonated |
| 9 | rc cola soda volume sweetener ` naturally ` ` flavours juice content type carbonization carbonated (20%) | rc cola cola soda rc_cola_type cola_type rc_cola_type_material carbonated_rc_cola_type volume_591 carbonated |
| 10 | ` ` water carbonated ` water volume carbonization carbonated caffeine naturally derived natural material type metal water type spring juice ` (19%) | premier water natural_spring_water_carbonated spring_water_carbonated spring_water_carbonated_carbonated water_carbonated_carbonated_water water_carbo |
