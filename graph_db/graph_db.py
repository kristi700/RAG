from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

class NebulaHandler:
    def __init__(self, space_name, host='nebula-docker-compose-graphd-1', port=9669, user='root', password='nebula'):
        """
        Initializes the NebulaHandler object and establishes a connection to NebulaGraph.

        Args:
            space_name (str): The name of the space to initialize and switch to.
            host (str): Host address of the NebulaGraph server.
            port (int): Port of the NebulaGraph server.
            user (str): Username for NebulaGraph.
            password (str): Password for NebulaGraph.
        """
        self.space_name = space_name
        self.config = Config()
        self.connection_pool = ConnectionPool()
        if not self.connection_pool.init([(host, port)], self.config):
            raise Exception("Failed to initialize the connection pool.")
        self.session = self.connection_pool.get_session(user, password)
        self.session.execute(f"""CREATE SPACE {self.space_name}(partition_num=1, replica_factor=1, vid_type=INT64);""")
        self.session.execute(f"USE {self.space_name};")

    def switch_space(self, space_name):
        """Switches the working space in NebulaGraph."""
        self.space_name = space_name
        self.session.execute(f"USE {self.space_name}")

    def recreate_space(self):
        """Drops and recreates the space with the default schema."""
        query = f'''
            DROP SPACE IF EXISTS {self.space_name};
            CREATE SPACE {self.space_name}(partition_num=1, replica_factor=1, vid_type=INT64);
            USE {self.space_name};
            CREATE TAG entity(name string, description string);
            CREATE EDGE relationship(relationship string);
            CREATE TAG INDEX entity_name_index ON entity(name(20));
            CREATE EDGE INDEX relationship_index ON relationship(relationship(20));
        '''
        self.session.execute(query)

    def execute_query(self, query):
        """Executes a NebulaGraph query and returns the result."""
        try:
            result = self.session.execute(query)
            if result.error_code() != 0:
                raise Exception(f"Error executing query: {result.error_msg()}")
            return result
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

    def get_max_entity_idx(self):
        """Fetches the maximum entity ID."""
        query = f"""MATCH (v:entity) RETURN id(v) AS id ORDER BY id DESC LIMIT 1;"""
        result = self.execute_query(query)
        if result and result.rows():
            return int(result.rows()[0].values[0].get_iVal())
        return 0

    def insert_entity(self, entity_name, entity_description, entity_idx=None):
        """Inserts or updates an entity."""
        entity_idx = entity_idx or self.get_max_entity_idx() + 1

        entity_name = entity_name.replace("'", "\\'")
        entity_description = entity_description.replace("'", "\\'")

        query = f"""INSERT VERTEX entity(name, description) VALUES {entity_idx}:('{entity_name}', '{entity_description}');"""
        self.execute_query(query)
        return entity_idx

    def insert_relationship(self, src_entity_idx, dst_entity_idx, relationship):
        """Creates a relationship (edge) between two entities."""
        query = f"""INSERT EDGE relationship(relationship) VALUES {src_entity_idx} -> {dst_entity_idx}:('{relationship}');"""
        self.execute_query(query)

    def upsert_entity_relationship(self, src_name, src_description, dst_name, dst_description, relationship):
        """Upserts two entities and their relationship."""
        src_id = self.get_id_by_name(src_name) or self.insert_entity(src_name, src_description)
        dst_id = self.get_id_by_name(dst_name) or self.insert_entity(dst_name, dst_description)
        self.insert_relationship(src_id, dst_id, relationship)
        return src_id, dst_id

    def get_id_by_name(self, entity_name):
        """Retrieves the ID of an entity by its name."""
        query = f"""MATCH (v:entity) WHERE v.name == '{entity_name}' RETURN id(v) AS id;"""
        result = self.execute_query(query)
        if result and result.rows():
            return int(result.rows()[0].values[0].get_iVal())
        return None

    def get_full_graph(self):
        """Fetches all nodes and edges."""
        query = """MATCH p=(v:entity)-[r]->(v1:entity) RETURN p;"""
        result = self.execute_query(query)
        return result
   