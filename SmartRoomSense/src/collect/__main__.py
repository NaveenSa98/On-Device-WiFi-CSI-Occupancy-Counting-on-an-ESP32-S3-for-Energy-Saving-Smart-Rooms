"""One entry point for all collection commands:  uv run python -m src.collect <command> ...

  record   save one recording from the board to a CSV           (logger.py)
  check    judge recordings: PASS or FAIL                        (checker.py)
  live     live plot while recording                             (viewer.py)
  plot     activity plot of finished recordings                  (viewer.py)
  map      mean amplitude per subcarrier position (pilot 2A.6)   (viewer.py)

Add --help after a command to see its options, e.g.  uv run python -m src.collect record --help
"""
import sys

from src.collect import checker, logger, viewer

USAGE = __doc__


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "record":
        return logger.main(rest)
    if command == "check":
        return checker.main(rest)
    if command in ("live", "plot", "map"):
        return viewer.main([command] + rest)
    print(f"Unknown command '{command}'.\n{USAGE}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
