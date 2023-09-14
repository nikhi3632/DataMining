#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import final


class Vehicle(ABC):

  def __init__(self, make, model, year, color):
    self.make = make
    self.model = model
    self.year = year
    self.color = color
  # The @abstractmethod decorator is used in Python to define abstract methods in abstract base classes. 
  # An abstract method is a method that is declared but does not contain an implementation. 
  # This allows derived classes to implement the method in their own way, 
  # while still guaranteeing that the method exists and has a certain signature.

  # An abstract base class is a class that cannot be instantiated, but is meant to serve as a base class for other classes. 
  # It is used to define a common interface for a set of classes, without providing an implementation. 
  # The abstract base class defines a set of abstract methods that derived classes must implement in order to be considered "complete".
  @abstractmethod
  def start(self):
    pass

  @abstractmethod
  def stop(self):
    pass

  def __str__(self):
    return f"{self.color} {self.year} {self.make} {self.model}"

  @final
  def get_wheels(self):
    print(f"{self} has {self._wheels} wheels.")

  @final
  def set_wheels(self, wheels):
    self._wheels = wheels


class Car(Vehicle):

  def __init__(self, make, model, year, color, num_doors):
    super().__init__(make, model, year, color)
    self.num_doors = num_doors

  def start(self):
    print(f"{self} is starting with key ignition.")

  def stop(self):
    print(f"{self} is stopping with foot brake.")

  def open_trunk(self):
    print(f"{self} has opened the trunk.")

  # A staticmethod is a method that does not need access to the instance attributes and operates only on the class level. 
  # Staticmethods are usually used for utility functions that are related to the class but do not need to access any instance-specific data.
  @staticmethod
  def honk():
    print("Honk Honk!")

  @classmethod
  def class_name(cls, num_wheels):
    print(f"{cls.__name__} has {num_wheels} wheels.")


class Van(Vehicle):

  def __init__(self, make, model, year, color, num_seats):
    super().__init__(make, model, year, color)
    self.num_seats = num_seats

  def start(self):
    print(f"{self} is starting with push button.")

  def stop(self):
    print(f"{self} is stopping with hand brake.")

  def slide_door(self):
    print(f"{self} has opened the slide door.")

  @classmethod
  def class_name(cls, num_wheels):
    print(f"{cls.__name__} has {num_wheels} wheels.")


class Bike(Vehicle):

  def __init__(self, make, model, year, color, engine_size):
    super().__init__(make, model, year, color)
    self.engine_size = engine_size

  def start(self):
    print(f"{self} is starting with kick start.")

  def stop(self):
    print(f"{self} is stopping with hand brake.")

  def wheelie(self):
    print(f"{self} is doing a wheelie.")

  @classmethod
  def class_name(cls, num_wheels):
    print(f"{cls.__name__} has {num_wheels} wheels.")


class Scooter(Vehicle):

  def __init__(self, make, model, year, color, engine_size):
    super().__init__(make, model, year, color)
    self.engine_size = engine_size

  def start(self):
    print(f"{self} is starting with electric start.")

  def stop(self):
    print(f"{self} is stopping with foot brake.")

  def beep_horn(self):
    print(f"{self} is beeping the horn.")

  @classmethod
  def class_name(cls, num_wheels):
    print(f"{cls.__name__} has {num_wheels} wheels.")


# For abstraction, I have used the @abstractmethod decorator to define abstract methods start and stop in the Vehicle class. 
# These methods are implemented in the derived classes but are not implemented in the Vehicle class itself, 
# allowing me to define a common interface for all vehicles while leaving the implementation details to the specific vehicle types.

# For encapsulation, I have encapsulated the make, model, year, and color attributes in the Vehicle class, and provided access to them through the str method. 
# This prevents direct access to the attributes from outside the class and provides a clean interface for accessing the vehicle's properties.

# For polymorphism, I have defined the start and stop methods in the Vehicle class and overridden them in the derived classes. 
# This allows me to call the start and stop methods on any vehicle object regardless of its type, 
# and the appropriate implementation will be called based on the object's actual type.

# For inheritance, I have defined a class hierarchy where the derived classes Car, Van, Bike, and Scooter inherit from the base class Vehicle. 
# This allows me to define a common set of attributes and methods that all vehicles share, 
# while still allowing each vehicle type to have its own unique attributes and behavior.

def get_info():
  print("-------------------------------------------------------------------")
  car1 = Car("Toyota", "Corolla", 2020, "Silver", 4)
  car1.start()
  car1.honk()
  car1.set_wheels(4)
  car1.get_wheels()
  car1.class_name(car1._wheels)
  print("-------------------------------------------------------------------")
  van1 = Van("Honda", "Odyssey", 2021, "White", 7)
  van1.start()
  van1.slide_door()
  van1.set_wheels(4)
  van1.get_wheels()
  van1.class_name(van1._wheels)
  print("-------------------------------------------------------------------")
  bike1 = Bike("Harley Davidson", "Street Glide", 2022, "Black", 1868)
  bike1.start()
  bike1.wheelie()
  bike1.set_wheels(2)
  bike1.get_wheels()
  bike1.class_name(bike1._wheels)
  print("-------------------------------------------------------------------")
  scooter1 = Scooter("Honda", "Activa", 2019, "Red", 110)
  scooter1.start()
  scooter1.beep_horn()
  scooter1.set_wheels(2)
  scooter1.get_wheels()
  scooter1.class_name(scooter1._wheels)
  print("-------------------------------------------------------------------")

if __name__ == "__main__":
  get_info()
