import mysql.connector
from mysql.connector import Error
import pandas as pd

""" try:
    connection = mysql.connector.connect(
        host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
        database='ds50',
        user='ds50',
        password='AVNS_4ybSd0CoPKnCL5F',
        port = '25060')
    
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

except Error as e:
    print("Error while connecting to MySQL", e)

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed") """

def getFirst1000Books():
    connection = mysql.connector.connect(
        host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
        database='ds50',
        user='ds50',
        password='AVNS_4ybSd0CoPKnCL5F',
        port = '25060')

    if connection.is_connected():
        df = pd.read_sql("SELECT * FROM BOOK LIMIT 1000;", connection)
        connection.close()

    return df.to_dict('records')

def getFirst1000Authors():
    connection = mysql.connector.connect(
        host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
        database='ds50',
        user='ds50',
        password='AVNS_4ybSd0CoPKnCL5F',
        port = '25060')

    if connection.is_connected():
        df = pd.read_sql("SELECT * FROM AUTHOR LIMIT 1000;", connection)
        connection.close()

    return df.to_dict('records')

def getFirst1000Works():
    connection = mysql.connector.connect(
        host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
        database='ds50',
        user='ds50',
        password='AVNS_4ybSd0CoPKnCL5F',
        port = '25060')

    if connection.is_connected():
        df = pd.read_sql("SELECT * FROM WORK LIMIT 1000;", connection)
        connection.close()

    return df.to_dict('records')

def getFirst1000Interactions():
    connection = mysql.connector.connect(
        host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
        database='ds50',
        user='ds50',
        password='AVNS_4ybSd0CoPKnCL5F',
        port = '25060')

    if connection.is_connected():
        df = pd.read_sql("SELECT * FROM INTERACTION LIMIT 1000;", connection)
        connection.close()

    return df.to_dict('records')

def getFirst1000Reviews():
    connection = mysql.connector.connect(
        host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
        database='ds50',
        user='ds50',
        password='AVNS_4ybSd0CoPKnCL5F',
        port = '25060')

    if connection.is_connected():
        df = pd.read_sql("SELECT * FROM REVIEW LIMIT 1000;", connection)
        connection.close()

    return df.to_dict('records')

def getFirst1000Wrotes():
    connection = mysql.connector.connect(
        host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
        database='ds50',
        user='ds50',
        password='AVNS_4ybSd0CoPKnCL5F',
        port = '25060')

    if connection.is_connected():
        df = pd.read_sql("SELECT * FROM WROTE LIMIT 1000;", connection)
        connection.close()

    return df.to_dict('records')