
class CannonControl():

    def __init__(self, motor_driver1, motor_driver2, motor_driver3):
        """
        This is a class that provides an interface between the program and the motor 
        drivers. The parameters here are important.

        Parameters:
            motor_driver1: This is the moter driver class that corrosponds
            to the motor that controls left to right motion. This means that
            its plane of rotation is parallell to the ground

            motor_driver2: This is the driver that corosponds to the motor 
            that controls up and down movment of the cannon. This means it's 
            axis or rotation is perpendicular to the ground. 

            motor_driver3: this is the driver that controls the actual firing
            of the cannon.
        """

        self.horizontal_motor = motor_driver1
        self.vertical_motor = motor_driver2
        self.cannon_motor = motor_driver3

    def left(self):
        self.horizontal_motor.inc()
        return

    def right(self):
        self.horizontal_motor.dec()
        return

    def up(self):
        self.vertical_motor.inc()
        return

    def down(self):
        self.vertical_motor.dec()
        return

    def fire(self):
        self.cannon_motor.fire()
        return

class VehicleControl():

    def __init__(self):
        return






        
