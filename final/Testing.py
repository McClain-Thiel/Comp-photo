
#utils testing

#detection testing

#drivers testing

#interface testing
import keyboard

def main():
    for x in range(500000000000):
        if x % 2342 == 0:
            print(x)

        if keyboard.is_pressed('q'):
            print("Stopping on: ", x)
            break

if __name__ == "__main__":
    main()