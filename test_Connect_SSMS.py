import pyodbc

# Thông tin kết nối
server = 'DESKTOP-B86U75E'
database = 'Law_ChatBot_DB'

# Chuỗi kết nối với Windows Authentication
conn_string = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes"

# Kết nối
try:
    conn = pyodbc.connect(conn_string)
    print("Kết nối thành công!")
except Exception as e:
    print("Lỗi kết nối:", e)
