class ColorPrint():

    foreground = '\033[38;2;'
    background = '\033[48;2;'
    end = '\033[0m'

    red = (170, 0, 0)
    orange = (215, 120, 45)
    yellow = (245, 210, 70)
    green = (10, 160, 10)
    blue = (30, 80, 190)
    purple = (160, 40, 160)
    white = (255, 255, 255)
    black = (0, 0, 0)

    @staticmethod
    def __color(color):
        return f"{color[0]};{color[1]};{color[2]}"

    @staticmethod
    def print_error(sstr):
        print(f"{ColorPrint.foreground}{ColorPrint.__color(ColorPrint.red)}m{sstr}{ColorPrint.end}")

    @staticmethod
    def print_warning(sstr):
        print(f"{ColorPrint.foreground}{ColorPrint.__color(ColorPrint.orange)}m{sstr}{ColorPrint.end}")


    @staticmethod
    def print_success(sstr):
        print(f"{ColorPrint.foreground}{ColorPrint.__color(ColorPrint.green)}m{sstr}{ColorPrint.end}")

    @staticmethod
    def print_info(sstr):
        print(f"{ColorPrint.foreground}{ColorPrint.__color(ColorPrint.blue)}m{sstr}{ColorPrint.end}")

    @staticmethod
    def print_message(sstr):
        print(f"{ColorPrint.foreground}{ColorPrint.__color(ColorPrint.purple)}m{sstr}{ColorPrint.end}")