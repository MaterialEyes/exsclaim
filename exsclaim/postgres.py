import psycopg2
from configparser import ConfigParser

class Database():

    def __init__(self, name, configuration_file = "database.ini"):
        parser = ConfigParser()
        parser.read(configuration_file)
        db_params = {}
        if parser.has_section(name):
            for key, value in parser.items(name):
                db_params[key] = value
        else:
            db_params = {
                "host": "localhost",
                "database": name,
                "user": "postgres"
            }
        self.connection = psycopg2.connect(**db_params)
        self.cursor = self.connection.cursor()

    def query(self, sql, data=None):
        self.cursor.execute(sql, data)

    def query_many(self, sql, data):
        psycopg2.execute_values(self.cursor, sql, data)

    def copy_from(self, file, table):
        table_to_copy_command = {
            "exsclaim_app_article": "exsclaim_app_article",
            "exsclaim_app_figure": "exsclaim_app_figure",
            "exsclaim_app_subfigure": "exsclaim_app_subfigure",
            "exsclaim_app_scalebar": "exsclaim_app_scalebar",
            "exsclaim_app_scalebarlabel": "exsclaim_app_scalebarlabel(text,x1,y1,x2,y2,label_confidence,box_confidence,nm,scale_bar_id)",
            "exsclaim_app_subfigurelabel": "exsclaim_app_subfigurelabel(text,x1,y1,x2,y2,label_confidence,box_confidence,subfigure_id)"
        }


        with open(file, "r") as csv_file:
            self.cursor.copy_expert(
                "COPY {} FROM STDIN CSV".format(table_to_copy_command[table]),
                csv_file
            )

    def close(self):
        self.cursor.close()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def commit(self):
        self.connection.commit()