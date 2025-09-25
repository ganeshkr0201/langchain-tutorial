from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'ganesh'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(ge=0, le=10, default=7, description='A decimal value representing the cgpa of the student')


new_student = {
    "name": "rohit",
    "age": 36,
    "email": "abc@gmail.com",
    "cgpa": 7.98
}

student = Student(**new_student)

student_dict = dict(student)
print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)