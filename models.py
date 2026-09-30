"""List Foundry models without initializing anything on import."""
from llm import get_manager


if __name__ == "__main__":
    for model in get_manager().catalog.list_models():
        print(model.alias)
