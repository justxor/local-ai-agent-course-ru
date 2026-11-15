import unittest

from agent import execute, run, search_notes


class AgentTests(unittest.TestCase):
    def test_search_returns_source(self):
        self.assertEqual(search_notes("Север релиз")[0]["source"], "sever.md")
        self.assertEqual(search_notes("абракадабра"), [])

    def test_tool_validates_name_and_arguments(self):
        self.assertIn("error", execute({"name": "unknown", "arguments": {}}))
        self.assertIn("error", execute({"name": "search_notes", "arguments": {"query": 9}}))

    def test_tool_result_is_passed_back_to_model(self):
        def fake_chat(payload):
            last = payload["messages"][-1]
            if last["role"] == "user":
                return {"role": "assistant", "content": "", "tool_calls": [{
                    "function": {"name": "search_notes", "arguments": {"query": "Север"}},
                }]}
            self.assertEqual(last["role"], "tool")
            self.assertIn("sever.md", last["content"])
            return {"role": "assistant", "content": "18 января 2027 года"}

        self.assertEqual(run("Когда релиз?", request_chat=fake_chat),
                         ("18 января 2027 года", ["search_notes"]))

    def test_loop_stops(self):
        def looping_model(_):
            return {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "search_notes", "arguments": {"query": "Север"}},
            }]}

        with self.assertRaisesRegex(RuntimeError, "лимит"):
            run("Когда релиз?", request_chat=looping_model)


if __name__ == "__main__":
    unittest.main()
