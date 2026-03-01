from src import prompt_store


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_store, "PROMPTS_DIR", tmp_path / "prompts")
    prompt_store.save_prompt("v1", "hello")
    loaded = prompt_store.load_prompt("v1")
    assert loaded.prompt_text == "hello"
