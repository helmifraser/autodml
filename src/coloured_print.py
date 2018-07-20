bcolors = {
    'header': '\033[95m',
    'okblue': '\033[94m',
    'okgreen': '\033[92m',
    'warn': '\033[93m',
    'fail': '\033[91m',
    'endc': '\033[0m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    }


def printc(str, type='fail'):
    print(bcolors[type] + str + bcolors['endc'])
