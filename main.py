from core.db_setup import initialize_db
from core.utils import perform_test

def main():
    initialize_db()
    perform_test()

if __name__ == "__main__":
    main()
