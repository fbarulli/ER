from model_tracks.live_logs import WorkerLogs


def test_worker_logs_forward_partial_lines_once_and_separate_uploads(tmp_path, capsys):
    path = tmp_path/'text__worker.log'
    path.write_text('loss=0.5\nepoch=')
    logs = WorkerLogs(tmp_path, ['text'])
    logs.drain()
    assert capsys.readouterr().out == '[track/text] loss=0.5\n'
    with path.open('a') as handle:
        handle.write('2\r[dvc] push started\nfinal')
    logs.drain()
    assert capsys.readouterr().out == '[track/text] epoch=2\n[artifact/text] [dvc] push started\n'
    logs.drain()
    assert capsys.readouterr().out == ''
    logs.drain(final=True)
    assert capsys.readouterr().out == '[track/text] final\n'
