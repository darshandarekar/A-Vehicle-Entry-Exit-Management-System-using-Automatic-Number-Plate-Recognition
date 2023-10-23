import sys
from my_db_other import dbhelper

class vehicle_check:
    def __init__(self):
        #connect to the database
        self.db = dbhelper()
        self.menu()
        
    def menu(self):
        user_input=input("""
                   1.Add the vehical in data base:
                   2.Check if vehicle is already in our database:
                   """)
        if user_input=="1":
            self.register()
                
        elif user_input =="2":
            self.numberplate()
        
        else:
            sys.exit(1000)
            
                       
    def register(self):
        emp_id = int(input("Enter the employ id: "))
        name = input("Enter the employee name: ")
        phone_no= int(input("Enter the mobile number: "))
        vehicle_no = str(input("Enter the vehicle number: "))
        
        response = self.db.register(emp_id,name,phone_no,vehicle_no)
        
        if response:
            print("Registration successful")
        
        else:
            print("Registration faild")
        
        self.menu()
            
  
    def numberplate(self):
        reg_plate=input("Enter the vehicale number plate")
        
        self.db.search(reg_plate)
        
        
        #if data: # means no list with tuple will return
        #   print("Vehicle is not availble in our database")
        #   self.numberplate()
        
        #else:
        #    print(f"Good! is found{data[0][1]}")


if __name__ =="__main__":
    obj = vehicle_check()
