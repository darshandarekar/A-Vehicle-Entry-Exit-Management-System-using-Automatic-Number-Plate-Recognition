import mysql.connector
import sys

# db= mysql.connector.connect(host="localhost", user="root", database = "my_test", passwd = "behappy@187302")
# mycourser=db.cursor()

# # this will show availble databases
# mycourser.execute("show databases;")
# for i in mycourser:
#    print(i)

# creating database #NOte remove the databse from db before running below code
# mycourser.execute("CREATE DATABASE  my_test")

# to create a table in database
# mycourser.execute("""CREATE TABLE anpr (
#                     EMP_ID INT PRIMARY KEY,
#                     NAME  VARCHAR(250) NOT NULL,
#                     PHONE_NO INT NOT NULL,
#                     VECHICLE_NO VARCHAR(250)
#                     )""")


# to delete any row from table
# mycourser.execute("ALTER TABLE my_test.anpr DROP id")

    
#db.close() #closing the connection


class dbhelper:
    def __init__(self):
        
        try:
            self.conn=mysql.connector.connect(host="localhost", user="root", database="parking",  passwd = "1999") # alway check this one from line 13 
            self.mycursor = self.conn.cursor()
        
        except:
            print("Not able to connect to database")
            sys.exit(0) # 0 - exit code for db and its use to stop the code from execution
            
        else:
            print("Connection done")
            
    
    def register(self,emp_id,name,phone_no,vehicle_no):
        try:
            self.mycursor.execute("INSERT INTO anpr VALUES(%s,%s, %s,%s)",(emp_id,name,phone_no,vehicle_no))
            self.conn.commit() ## to enter the data into table from ram
        
        except:
            return -1
        else:
            return 1
        
    
    def search(self,reg_plate):  ## to read the record from DB 
        
        try:
            self.mycursor.execute("SELECT * FROM anpr WHERE vehicle_no = (%s)",(reg_plate,))
        
            data=self.mycursor.fetchall() # basically we are getting all stored vale from mycursor
            for x in data:
                print(x)
        
        finally:
            self.conn.close() # disconnecting from server

            