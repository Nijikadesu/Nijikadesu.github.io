---
authors:
    - mingkun
categories:
    - Python
date: 2024-05-08
tags:
    - 类与对象
slug:  "cls&obj"
---
# Python类与对象浅析

我在学习李沐动手学习深度学习的过程中，在学习层与块的概念时，对于实现自定义块的代码中的super函数的具体功能产生疑问，查找资料后做此记录。

<!-- more -->

## Python面向对象
### 何为面向对象
在学习C语言的时候，我们倾向于将解决问题的过程显式地体现在一系列函数与数据中，通过按照特定顺序执行函数达到解决问题的效果。

在面向过程中，我们定义的函数和数据往往离散的分布在代码的各处，这大大降低了代码的可读性和扩展性。对代码重用性，灵活性和扩展性的需求催生了面向对象编程架构的诞生，面向对象把事物抽象成对象的概念，就是说这个问题里面有哪些对象，然后给对象赋一些属性和方法，然后让每个对象去执行自己的方法，问题得到解决。

类与对象是面向对象程序设计的核心概念。

- 类(Class)： 用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。

类是对象的抽象，对象是类的具体。Python将一系列属性和方法封装在了名为类的容器中，并通过实例化对象提供访问类成员的接口。
### 为什么要面向对象
面向对象具有易维护、易复用、易扩展的特点，由于面向对象有封装、继承、多态性的特性，可以设计出低耦合的系统，使系统 更加灵活、更加易于维护。相对的，面向对象相比面向过程（以C为代表）有着较低的性能表现。
### 如何面向对象
**类继承**
在Python中，我们使用class语句创建一个新类
```python
class Emplooyee:
	'a class including all emploees'
	empCount = 0 # 类变量，在所有实例中共享
	    
	def _init_(self, name, salary):
		self.name = name
		self.salary = salary
		Employee.empCount += 1
		
	def displayCount(self):
		print "Total employee %d" % Employee.empCount

	def dispalyEmployee(self):
		print "Name: ", self.name, ", Salary", self.salary
```
- `empCount`为类变量，使用`Employee.empCount`访问
- `__init__()`方法称为类构造函数，在创建实例时会自动调用此方法
- `self`代表类的实例，在定义类方法时必须出现在参数表头

你可以使用类的名称进行实例化，并使用`.`来访问对象的属性
```python
# create an object for class Employee
emp1 = Employee("Zara", 2000)
emp2 = Employee("Sam", 3000)
# vist a member
emp1.displayEmployee()
emp2.displayEmployee()
print "Total employee %d" % Employee.empCount
# 输出结果如下：
# Name :  Zara ,Salary:  2000
# Name :  Manni ,Salary:  5000
# Total Employee 2
# 可以看出，类变量不会随着新对象的实例化而重新生成，他是静态的
```
你也可以添加，删除，修改类属性
```python
emp1.age = 7  # 添加一个 'age' 属性
emp1.age = 8  # 修改 'age' 属性
del emp1.age  # 删除 'age' 属性
```
通过调用析构函数`__del__()`实现对象的销毁
```python
a = 40 # 声明了数据对象a
del a # 销毁a
```
你也可以在类中自定义析构函数
```python
class Point: 
	def __init__( self, x=0, y=0): 
		self.x = x 
		self.y = y 
	
	def __del__(self): 
		class_name = self.__class__.__name__ 
		print class_name, "销毁" 
		
pt1 = Point() 
pt2 = pt1 
pt3 = pt1 
print id(pt1), id(pt2), id(pt3) # 打印对象的id 
del pt1 
del pt2 
del pt3
# 输出结果：
#3083401324 3083401324 3083401324
# Point 销毁
# pt2与pt3是对pt1的引用，他们对应同一片内存空间，在销毁pt1时，# pt2与pt3同步销毁
```
**类继承**
类的继承机制是面向对象中实现代码重用的方法之一
```python
# 类继承基础语法
class 派生类名(基类名)
    ...
```
如果子类不重写父类的构造方法，实例化时会自动调用父类构造方法
```python
class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name) )
    def getName(self):
        return 'Father ' + self.name
 
class Son(Father):
    def getName(self):
        return 'Son '+self.name
 
if __name__=='__main__':
# 当python文件作为主程序运行时表达式为1，作为外部库import时为0
    son=Son('runoob')
    print ( son.getName() )
# 输出
# name: runoob
# Son runoob
```
如果子类重写父类构造方法，实例化时仅调用子类构造方法
```python
class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name) )
    def getName(self):
        return 'Father ' + self.name
 
class Son(Father):
    def __init__(self, name):
        print ( "hi" )
        self.name =  name
    def getName(self):
        return 'Son '+self.name
 
if __name__=='__main__':
    son=Son('runoob')
    print ( son.getName() )
# 输出
# hi
# Son runoob
```
如果子类重写构造方法，亦要继承父类构造方法，使用`super`关键字
```python
class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name))
    def getName(self):
        return 'Father ' + self.name
 
class Son(Father):
    def __init__(self, name):
        super(Son, self).__init__(name)
        # 注：python支持super()方法省略参数传递
        # 与super()._init_(name)有相同功能
        print ("hi")
        self.name =  name
    def getName(self):
        return 'Son '+self.name
 
if __name__=='__main__':
    son=Son('runoob')
    print ( son.getName() )
# 输出
# name: runoob
# hi
# Son runoob
```
## super函数与实例构造过程及其在PyTorch中的应用
### super函数的基础用法

在Python中，我们通常使用`super()`函数来引用父类。基础的用法是带有两个参数：第一个是子类，第二个是对象，即`super(subclass, instance)。`其用途是返回一个临时对象，该对象绑定到父类的方法上，而不是子类的方法。

```python
class Parent():
    def hello(self):
        print("Hello from Parent")

class Child(Parent):
    def hello(self):
        super(Child, self).hello()
        print("Hello from Child")

c = Child()
c.hello()
# 输出
# Hello from Parent
# Hello from Child
```

本质上，`super(Child, self).hello()`就等于`Parent.hello(self)`。使用`super`的好处是不必显式写出父类具体是哪一个，这样有利于后续维护与更新（比如改变了父类，但是这段代码不用改）。

#### super在类定义中的特殊用法

在类的定义中，`super`函数的使用可以更加简化。我们可以不必提供任何参数，Python解释器会自动填充参数。

```python
class Parent():
    def hello(self):
        print("Hello from Parent")

class Child(Parent):
    def hello(self):
        super().hello()
        print("Hello from Child")

c = Child()
c.hello()
```

这段代码的输出与前一段代码相同。Python解释器在执行`super().hello()`时，会自动填充当前类和实例。也就是说，在类定义中，`super()`等价于`super(Child, self)`。实际上解释器使用的是`__class__`变量，每个函数中都能访问这个变量，它指向这个函数对应的类。

### init & new 与实例化过程  

Python的`__init__`方法在面向对象编程中非常重要。它是一种特殊的方法，用于在创建对象时初始化对象的状态。然而，需要注意的是，`__init__`方法并不是实例化一个对象的过程，它仅仅是初始化对象状态的一部分。

实际上，真正的实例化过程是由`__new__`方法完成的。`__new__`方法是一个静态方法，用于创建并返回一个对象。在Python中，调用`Class()`实际上包含两个步骤：首先调用`__new__`方法创建一个新对象，然后调用`__init__`方法来初始化该对象。

```python
class MyClass():
    def __new__(cls):
        print("Creating a new object")
        return super().__new__(cls)

    def __init__(self):
        print("Initializing the object")

m = MyClass()
```

这段代码会输出：

```python
Creating a new object
Initializing the object
```

通过这段代码，我们可以看到实例化过程实际上包括`__new__`和`__init__`两个步骤。

###  `super().__init__()`

在使用PyTorch构建神经网络模型时，我们通常需要自定义模型类并继承PyTorch的基础模块类`nn.Module`。在自定义的模型类中，通常需要在`__init__`方法中调用`super().__init__()`，这是为了正确地**初始化`nn.Module`类的内部状态**。只有调用了`super().__init__()`之后，才能创建子模块：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        # 下面两行代码，交换顺序就会报错
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = MyModel(3, 5)
```