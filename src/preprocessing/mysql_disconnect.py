def disconnect(cnx=None):
    try:
        cnx.close()
    except:
        print("Something went wrong. Do not run this module in standalone mode.")

if __name__ == "__main__":
    print("Running disconnect module of RXNormVisualize package.")
    disconnect()