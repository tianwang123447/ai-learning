def get_level(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D"

def greet(name, msg="你好"):
    print(msg, name)

print(get_level(85))
greet("田旺")
greet("田旺", "早上好")


def get_level(score):
    if score >90:
        return "A"
    elif score > 80:
        return "B"
    elif score > 60:
        return "C"
    else:
        return "D"
    
def greet(name,msg="你好"):
        print(msg,name)

print(get_level(85))
greet("田旺")
greet("田旺","早上好")
