from fastapi.testclient import TestClient

from experiment_lab.api import app
from experiment_lab.dashboard import render_dashboard
from experiment_lab.decision import analyse
from experiment_lab.core import simulate


def test_api_status_and_simulation():
    client = TestClient(app)
    assert client.get('/health').json()['mode'] == 'synthetic_only'
    assert client.post('/simulate', json={'population': 1}).status_code == 422
    response = client.post('/simulate', json={'population': 3000, 'seed': 12, 'effect': 0.05})
    assert response.status_code == 200
    assert response.json()['simulation_decision']['production_launch_authorised'] is False


def test_static_dashboard_export(tmp_path):
    source = __import__('pathlib').Path(__file__).resolve().parents[2] / 'data' / 'synthetic_user_reports.csv'
    result = analyse(simulate(source, population=3000, seed=15))
    output = tmp_path / 'index.html'
    render_dashboard(result, output)
    page = output.read_text(encoding='utf-8')
    assert '<html lang="en">' in page
    assert 'PRODUCTION LAUNCH: NEVER AUTHORISED' in page
    assert 'Estimated SLA success by arm' in page


def test_dashboard_interaction_controls_and_policy_audit(tmp_path):
    source = __import__('pathlib').Path(__file__).resolve().parents[2] / 'data' / 'synthetic_user_reports.csv'
    report = analyse(simulate(source, population=3000, seed=51))
    assert report['routing_audit']['passed'] is True
    assert report['weight_diagnostics']['control']['n_observed'] > 0
    output = tmp_path / 'index.html'
    render_dashboard(report, output)
    page = output.read_text(encoding='utf-8')
    assert 'id="scenario-size"' in page
    assert 'id="scenario-effect"' in page
    assert 'id="sandbox-run"' in page
    assert 'Simulation-only statistical checks' in page
