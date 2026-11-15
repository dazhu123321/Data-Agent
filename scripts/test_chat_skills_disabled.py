"""Offline regression checks for the retail app's disabled Skills policy.

Run: .venv/Scripts/python.exe scripts/test_chat_skills_disabled.py
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask
from openai.types.chat import ChatCompletionChunk
from agent.activation import ActivationContext
from agent.agent import BusinessAgent
from agent.tools.exposure import filter_tools_for_turn
from agent.tools.schemas import AGENT_TOOLS
from api.chat import ActivationRequestError, _resolve_activation
from api.skills import bp as skills_blueprint


class DisabledSkillsTests(unittest.TestCase):
    def test_catalog_returns_no_skills_without_loading_files(self):
        app = Flask(__name__)
        app.register_blueprint(skills_blueprint)
        with patch('api.skills.SkillLoader') as loader:
            response = app.test_client().get('/api/skills?mode=business')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['skills'], [])
        self.assertFalse(response.json['enabled'])
        loader.assert_not_called()

    def test_explicit_api_activation_is_rejected(self):
        with self.assertRaises(ActivationRequestError) as caught:
            _resolve_activation(SimpleNamespace(session_id='test'), {'skill': 'kmeans'})
        self.assertEqual(caught.exception.code, 'skills_disabled')

    def test_restored_followup_skill_is_rejected_and_consumed(self):
        session = SimpleNamespace(session_id='test', auto_loaded_skill='kmeans')
        with self.assertRaises(ActivationRequestError):
            _resolve_activation(session, {}, allow_session_skill=True)
        self.assertEqual(session.auto_loaded_skill, '')

    def test_direct_agent_activation_cannot_bypass_api(self):
        agent = BusinessAgent.__new__(BusinessAgent)
        events = list(agent.run('聚类', [], activation=ActivationContext(skill_name='kmeans')))
        self.assertEqual([e['type'] for e in events], ['error', 'done'])
        self.assertIn('禁用 Skills', events[0]['message'])
        self.assertIsNone(agent._get_skill_def('kmeans'))

    def test_skill_tool_hidden_even_if_previously_discovered(self):
        tools = filter_tools_for_turn(
            AGENT_TOOLS, user_message='使用技能分析销售数据',
            discovered_tools=['load_analysis_skill'], has_data_source=True,
        )
        names = {t['function']['name'] for t in tools}
        self.assertNotIn('load_analysis_skill', names)
        self.assertIn('query_data', names)
        self.assertIn('query_knowledge', names)

    def test_normal_chat_ignores_client_auto_match_flag(self):
        client = MagicMock()
        client.chat.completions.create.return_value = iter([
            ChatCompletionChunk(id='offline-test', created=0, model='gpt-4o-mini',
                object='chat.completion.chunk', choices=[{'index': 0,
                'delta': {'content': '你好，请输入业务问题。'}, 'finish_reason': None}]),
            ChatCompletionChunk(id='offline-test', created=0, model='gpt-4o-mini',
                object='chat.completion.chunk', choices=[{'index': 0,
                'delta': {}, 'finish_reason': 'stop'}]),
        ])
        # The offline test must not start the unrelated MCP event loop.
        with patch('agent.agent.get_mcp_manager', return_value=MagicMock()):
            agent = BusinessAgent(client=client, model='gpt-4o-mini', workspace_id='')
        with patch('agent.agent.SkillLoader') as loader, \
             patch('agent.agent.load_instruction_section', return_value=''):
            events = list(agent.run('你好', [], auto_match_skill=True, memory_enabled=False))
        loader.assert_not_called()
        client.chat.completions.create.assert_called()
        request = client.chat.completions.create.call_args.kwargs
        self.assertNotIn('load_analysis_skill', {t['function']['name'] for t in request['tools']})
        self.assertTrue(any(e['type'] == 'text' and '你好' in e['content'] for e in events))


if __name__ == '__main__':
    unittest.main(verbosity=2)
