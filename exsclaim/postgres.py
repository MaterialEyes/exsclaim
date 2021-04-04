import psycopg2
from psycopg2 import sql
from configparser import ConfigParser

def initialize_database():
    conn = psycopg2.connect(
        database="postgres",
        user='postgres',
        password='password',
        host='127.0.0.1',
        port= '5432'
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("""CREATE DATABASE exsclaim""")
    conn.close()


class Database():

    def __init__(self, name, configuration_file = "database.ini"):
        try:
            initialize_database()
        except Exception as e:
            pass
        parser = ConfigParser()
        parser.read(configuration_file)
        db_params = {}
        if parser.has_section(name):
            for key, value in parser.items(name):
                db_params[key] = value
        else:
            db_params = {
                "host": "localhost",
                "database": "exsclaim",
                "user": "postgres"
            }
        self.connection = psycopg2.connect(**db_params)
        self.cursor = self.connection.cursor()

    def query(self, sql, data=None):
        self.cursor.execute(sql, data)

    def query_many(self, sql, data):
        psycopg2.execute_values(self.cursor, sql, data)

    def copy_from(self, file, table):
        app_name = "results"
        table_to_copy_command = {
            app_name + "_article": app_name + "_article_temp",
            app_name + "_figure": app_name + "_figure_temp",
            app_name + "_subfigure": app_name + "_subfigure_temp",
            app_name + "_scalebar": app_name + "_scalebar_temp",
            app_name + "_scalebarlabel": (
                app_name + "_scalebarlabel_temp(text,x1,y1,x2,y2,label_confidence,box_confidence,nm,scale_bar_id)"
            ),
            app_name + "_subfigurelabel": (
                app_name + "_subfigurelabel_temp(text,x1,y1,x2,y2,label_confidence,box_confidence,subfigure_id)"
            )
        }
        table_name = table
        temp_name = table_name + "_temp"
        # create a temporary table to copy data into. We then use copy to
        # populate table with a csv contents. Then we insert temp table
        # contents into the real table, ignoring conflicts. We use copy
        # because it is faster than insert, but create the temporary table
        # to mimic a nonexistent "COPY... ON CONFLICT" command
        self.query(
            sql.SQL(
                "CREATE TEMPORARY TABLE {} (LIKE {} INCLUDING ALL) ON COMMIT DROP;"
            ).format(sql.Identifier(temp_name), sql.Identifier(table_name))
        )
        with open(file, "r") as csv_file:
            self.cursor.copy_expert(
                "COPY {} FROM STDIN CSV".format(table_to_copy_command[table]),
                csv_file
            )
        self.query(
            sql.SQL(
                "INSERT INTO {} SELECT * FROM {} ON CONFLICT DO NOTHING;"
            ).format(sql.Identifier(table_name), sql.Identifier(temp_name))
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

if __name__ == "__main__":
    initialize_database()