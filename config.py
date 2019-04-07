import configparser


class Parser:
    def __init__(self):
        self.parser = configparser.ConfigParser()
        self.parser.read("params.ini")

    def get_section(self, section_name):
        return dict(self.parser.items(section_name))

    def get_sections(self, section_names):
        args = {}
        for section in section_names:
            args.update(self.get_section(section))
        return args
