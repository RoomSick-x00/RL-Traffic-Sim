"""Compatibility entry point for the shared DQN training configuration."""

if __package__:
    from .training import main
else:
    from training import main


if __name__ == "__main__":
    main()
