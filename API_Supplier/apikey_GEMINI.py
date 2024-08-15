class APIKeyManager:
    def __init__(self, keys):
        self.keys = {key: 0 for key in keys}  # Khởi tạo key với số lần sử dụng bằng 0
        self.index = 0
        self.key_list = list(self.keys.keys())

    def get_next_key(self):
        # Kiểm tra nếu tất cả các key đều có số lần sử dụng bằng nhau
        if len(set(self.keys.values())) == 1:
            self.index = 0  # Đặt lại chỉ số

        key = self.key_list[self.index]
        self.keys[key] += 1  # Tăng số lần sử dụng của key
        self.index = (self.index + 1) % len(self.key_list)
        return key

    def get_key_usage(self):
        return self.keys

if __name__ == "__main__":

    # Khởi tạo danh sách API keys
    api_keys = [
        'AIzaSyAJqkq0C4EBjcXFbcWtzaD3TnGwk9QEmaw',
        'AIzaSyB-2u84ae92PQvhBwqeWGfWfJKuoLsNp4E',
        'AIzaSyCWtPpwjXYIKAVrMH5Uf2PRpZNspaZu6ps',
        'AIzaSyDSvpg-rqX4ulYnYSmZTCSHr0i7lONIx_4'
    ]

    # Khởi tạo APIKeyManager với danh sách keys
    key_manager = APIKeyManager(api_keys)

    # Thực hiện 15 lần yêu cầu API
    for _ in range(15):
        response_data = key_manager.get_next_key()
        print(response_data)
    # In ra số lần sử dụng của mỗi key
    print(key_manager.get_key_usage())
