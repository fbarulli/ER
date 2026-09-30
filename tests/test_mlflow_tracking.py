import pytest


def test_sqlite_tracking_keeps_nested_metrics_and_artifacts(tmp_path, monkeypatch):
    mlflow = pytest.importorskip('mlflow')
    from core import mlflow_ctx
    monkeypatch.setenv('MLFLOW_TRACKING_URI', '')
    monkeypatch.setattr(mlflow_ctx, '_MLRUNS', tmp_path/'tracking')
    previous_uri = mlflow.get_tracking_uri()
    artifact = tmp_path/'report.txt'
    artifact.write_text('training report')
    try:
        with mlflow_ctx.MlflowCtx('skinny-tracking-check') as tracking:
            tracking.log_params({'track':'text'})
            tracking.log_metrics({'loss':0.25, 'invalid':float('nan')})
            tracking.log_artifact(artifact)
            parent = tracking.parent.info.run_id
            with tracking.nested as child:
                tracking.log_metrics({'dev_pr_auc':0.75})
                child_id = child.info.run_id
        client = mlflow.tracking.MlflowClient()
        assert client.get_run(parent).data.metrics == {'loss':0.25}
        assert client.get_run(parent).data.params['track'] == 'text'
        assert client.get_run(child_id).data.tags['mlflow.parentRunId'] == parent
        assert client.get_run(child_id).data.metrics['dev_pr_auc'] == 0.75
        assert client.list_artifacts(parent)[0].path == 'report.txt'
    finally:
        mlflow.set_tracking_uri(previous_uri)
