from chess_gnn.configuration import LocalHydraConfiguration
from chess_gnn.tasks.base import Task, get_config_path


def entrypoint(task_name: str):
    try:
        config_path = get_config_path(task_name)
        config = LocalHydraConfiguration(config_path)
    except FileNotFoundError:
        print(f"Task '{task_name}' not found")
        return

    task: Task = config.resolve(list(config.cfg.keys())[0])
    print(f"Running task {type(task).__name__}")
    task.run(configuration_path=config_path)


if __name__ == '__main__':
    entrypoint('WriteH5')
