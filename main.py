import sys
import ast

from session import Session


def main():
    agents = ast.literal_eval(sys.argv[1])
    print(agents)
    session = Session(2000, 2, 500)
    try:
        session.run(agents)
    except IndexError:
        print("No agents were specified.")


if __name__ == '__main__':
    main()
