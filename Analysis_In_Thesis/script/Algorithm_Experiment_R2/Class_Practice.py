# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:23:47 2024

@author: yisss
"""
##################################################################################
# class Student(object):
#     def __init__(self, name, age, gender, level, grades=None):
#         self.name = name
#         self.age = age
#         self.gender = gender
#         self.level = level
#         self.grades = grades or {}

#     def setGrade(self, course, grade):
#         self.grades[course] = grade

#     def getGrade(self, course):
#         return self.grades[course]

#     def getGPA(self):
#         return sum(self.grades.values())/len(self.grades)
    
# # 定义一些学生
# john = Student("John", 12, "male", 6, {"math":3.3})

# jane = Student("Jane", 12, "female", 6, {"math":3.5})

# # 现在我们可以很容易地得到分数
# print(john.getGrade("math"))
# print(john.getGPA())
# print(jane.getGPA())

##################################################################################
# class person(object):
#     address = '中国'  # 类属性，没个实例的公共属性

#     def __init__(self, name, sex, age):  # 相当于java中的构造方法
#         self.name = name  # 实例属性
#         self.sex = sex  # 实例属性
#         self.age = age  # 实例属性

#     def dance(self):  # 方法
#         print(self.name, '跳了一场舞')

# hong = person('小红', '女', 18)  # 实例化小红，将实例化的对象赋值给变量hong
# ming = person('小明', '男', 26)
# hua = person('小花', '女', 22)

# hua.dance()

###################################################################################

# class Circle(object):
#    __pi = 3.14

#    def __init__(self, r):
#        self.r = r

#    def area(self):
#        """
# 圆的面积
#        """
#        return self.r **2* self.__pi

# circle1 = Circle(1)
# print(Circle.__pi)  # 抛出AttributeError异常
# print(circle1.__pi)  # 抛出AttributeError异常

##################################################################################

class Animal:  #  python3中所有类都可以继承于object基类
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def call(self):
        print(self.name, '会叫')
       
class Cat(Animal):
    def __init__(self,name,age,sex):
        super(Cat, self).__init__(name,age)  # 不要忘记从Animal类引入属性 
        self.sex=sex                         # 注意：一定要用 super(Cat, self).__init__(name,age) 去初始化父类，
                                             #否则，继承自 Animal的 Cat子类将没有 name和age两个属性。

    def call(self):
       print(self.name, '会“喵喵”叫')
       
       
# if __name__ == '__main__':  # 单模块被引用时下面代码不会受影响，用于调试
#     c = Cat('喵喵', 2, '男')  #  Cat继承了父类Animal的属性   # '喵喵' 是name
#     c.call()  # 输出 喵喵 会叫 ，Cat继承了父类Animal的方法 
    
class Dog(Animal):
    def __init__(self, name, age, sex):
        super(Dog, self).__init__(name, age)
        self.sex = sex
        
    def call(self):
        print(self.name, '会“汪汪”叫')
        

def do(all):
    all.call()
    
A = Animal('小黑', 4)
C = Cat('喵喵', 5,'男')
D = Dog('蹦蹦', 2, '男')

for x in (A, C, D):
    do(x)
    


