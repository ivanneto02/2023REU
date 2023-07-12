import mysql.connector
from config import *

def connect(user=None, password=None, host=None, database=None):
    import mysql.connector
    cnx = mysql.connector.connect(
        user = user,
        password = password,
        host = host,
        database = database)
    return cnx

if __name__ == "__main__":
    print("Running connect module of RXNormVisualize package.")