UNREADABLE = "We couldn't read this file as audio"


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.error = {'code': code, 'message': message}
