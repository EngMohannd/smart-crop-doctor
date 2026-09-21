# Author: Prof. MM Ghassemi <ghassem3@msu.edu>
# Updated for Homework 2: Vector Embeddings & Programmatic Generation

from .embeddings import generate_embedding
import psycopg2
import psycopg2.extras
import glob
import json
import csv
from io import StringIO
import itertools
import hashlib
import os
import cryptography
from cryptography.fernet import Fernet
from math import pow
from flask import current_app


class database:
    """
    Database management class for PostgreSQL operations.
    
    Handles database connections, table creation, data insertion,
    and user authentication with encryption support and vector embeddings.
    """

    def __init__(self, purge=False):
        """
        Initialize database connection and configuration.
        
        Args:
            purge (bool): Whether to purge existing tables
        """
        # Grab information from the configuration file
        self.database   = current_app.config.get('DATABASE_NAME')
        self.host       = current_app.config.get('DATABASE_HOST')
        self.user       = current_app.config.get('DATABASE_USER')
        self.port       = current_app.config.get('DATABASE_PORT')
        self.password   = current_app.config.get('DATABASE_PASSWORD')
        
        # Tables must be created in order due to foreign key constraints
        self.tables = ['institutions', 'positions', 'experiences', 'skills', 'users']

        # Encryption configuration
        self.encryption = {
            'oneway': {
                'salt': current_app.config.get('ENCRYPTION_ONEWAY_SALT').encode(),
                'n': current_app.config.get('ENCRYPTION_ONEWAY_N'),
                'r': current_app.config.get('ENCRYPTION_ONEWAY_R'),
                'p': current_app.config.get('ENCRYPTION_ONEWAY_P')
            },
            'reversible': { 'key': current_app.config.get('ENCRYPTION_REVERSIBLE_KEY')}
        }

    #--------------------------------------------------
    # DATABASE QUERY FUNCTION
    #--------------------------------------------------
    def query(self, query="SELECT * FROM users", parameters=None):
        """
        Execute a database query with optional parameters.
        
        Args:
            query (str): SQL query to execute
            parameters (tuple, optional): Query parameters for parameterized queries
            
        Returns:
            list: Query results as list of dictionaries
        """
        # Establish database connection
        cnx = psycopg2.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port,
            database=self.database
        )

        # Execute query with or without parameters
        cur = cnx.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if parameters is not None:
            cur.execute(query, parameters)
        else:
            cur.execute(query)

        # Only fetch results for SELECT/SHOW queries
        row = []
        if query.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
            row = cur.fetchall()
        
        # Commit transaction and close connections
        cnx.commit()
        cur.close()
        cnx.close()
        
        return row

    #--------------------------------------------------
    # RESUME DATA FUNCTIONS
    #--------------------------------------------------
    def getResumeData(self):
        data = {}

        # For each institution
        institutions = self.query("SELECT * FROM institutions")
        for i in institutions:
            institution_id = i['inst_id']
            data[institution_id] = i
            data[institution_id].pop('inst_id', None)

            # For each position
            positions = self.query(f"""SELECT * FROM positions WHERE inst_id = {institution_id} ORDER BY start_date DESC""")
            data[institution_id]['positions'] = {}
            for p in positions:
                position_id = p['position_id']
                data[institution_id]['positions'][position_id] = p
                data[institution_id]['positions'][position_id].pop('position_id', None)
                data[institution_id]['positions'][position_id].pop('inst_id', None)

                # For each experience
                experiences = self.query(f"""SELECT * FROM experiences WHERE position_id = {position_id} ORDER BY start_date DESC""")
                data[institution_id]['positions'][position_id]['experiences'] = {}
                for e in experiences:
                    experience_id = e['experience_id']
                    data[institution_id]['positions'][position_id]['experiences'][experience_id] = e
                    data[institution_id]['positions'][position_id]['experiences'][experience_id].pop('experience_id', None)
                    data[institution_id]['positions'][position_id]['experiences'][experience_id].pop('position_id', None)

                    # For each skill
                    skills = self.query(f"""SELECT * FROM skills WHERE experience_id = {experience_id}""")
                    data[institution_id]['positions'][position_id]['experiences'][experience_id]['skills'] = {}
                    for s in skills:
                        skill_id = s['skill_id']
                        data[institution_id]['positions'][position_id]['experiences'][experience_id]['skills'][skill_id] = s
                        data[institution_id]['positions'][position_id]['experiences'][experience_id]['skills'][skill_id].pop('experience_id', None)
                        data[institution_id]['positions'][position_id]['experiences'][experience_id]['skills'][skill_id].pop('skill_id', None)

        # Convert dates to simple strings before returning
        for inst in data.values():
            for pos in inst.get('positions', {}).values():
                if pos.get('start_date'):
                    pos['start_date'] = str(pos['start_date'])[:7]
                if pos.get('end_date') and pos['end_date'] != '0000-00-00':
                    pos['end_date'] = str(pos['end_date'])[:7]
                else:
                    pos['end_date'] = 'Present'
                    
                for exp in pos.get('experiences', {}).values():
                    if exp.get('start_date'):
                        exp['start_date'] = str(exp['start_date'])[:7]
                    if exp.get('end_date') and exp['end_date'] != '0000-00-00':
                        exp['end_date'] = str(exp['end_date'])[:7]
                    else:
                        exp['end_date'] = ''
        
        return data

    #--------------------------------------------------
    # VECTOR EMBEDDING GENERATION
    #--------------------------------------------------
    def populate_initial_embeddings(self):
        """Populate embedding vectors for rows that do not have them yet"""
        try:
            # 1. Institutions: name + department
            institutions = self.query("SELECT inst_id, name, COALESCE(department, '') as dept FROM institutions WHERE embedding IS NULL")
            if institutions:
                for inst in institutions:
                    text = f"{inst['name']} {inst['dept']}".strip()
                    emb = generate_embedding(text)
                    self.query("UPDATE institutions SET embedding = %s WHERE inst_id = %s", (emb, inst['inst_id']))

            # 2. Positions: title + responsibilities
            positions = self.query("SELECT position_id, title, COALESCE(responsibilities, '') as resp FROM positions WHERE embedding IS NULL")
            if positions:
                for pos in positions:
                    text = f"{pos['title']} {pos['resp']}".strip()
                    emb = generate_embedding(text)
                    self.query("UPDATE positions SET embedding = %s WHERE position_id = %s", (emb, pos['position_id']))

            # 3. Experiences: name + description
            experiences = self.query("SELECT experience_id, name, COALESCE(description, '') as descr FROM experiences WHERE embedding IS NULL")
            if experiences:
                for exp in experiences:
                    text = f"{exp['name']} {exp['descr']}".strip()
                    emb = generate_embedding(text)
                    self.query("UPDATE experiences SET embedding = %s WHERE experience_id = %s", (emb, exp['experience_id']))

            # 4. Skills: name
            skills = self.query("SELECT skill_id, name FROM skills WHERE embedding IS NULL")
            if skills:
                for sk in skills:
                    text = f"{sk['name']}".strip()
                    emb = generate_embedding(text)
                    self.query("UPDATE skills SET embedding = %s WHERE skill_id = %s", (emb, sk['skill_id']))

            # 5. Users: email
            users = self.query("SELECT user_id, email FROM users WHERE embedding IS NULL")
            if users:
                for u in users:
                    text = f"{u['email']}".strip()
                    emb = generate_embedding(text)
                    self.query("UPDATE users SET embedding = %s WHERE user_id = %s", (emb, u['user_id']))

            print("[Database] Successfully populated vector embeddings for tables.")
        except Exception as e:
            print(f"[Database] Error populating embeddings: {str(e)}")

    #--------------------------------------------------
    # TABLE CREATION
    #--------------------------------------------------
    def createTables(self, purge=False, data_path='flask_app/database/'):
        if purge:
            for table in self.tables[::-1]:
                self.query(f"""DROP TABLE IF EXISTS {table} CASCADE""")
            
        # Execute all SQL queries in the /database/create_tables directory.
        for table in self.tables:
            with open(data_path + f"create_tables/{table}.sql") as read_file:
                create_statement = read_file.read()
            self.query(create_statement)

            # Import the initial data
            try:
                params = []
                with open(data_path + f"initial_data/{table}.csv") as read_file:
                    scsv = read_file.read()            
                for row in csv.reader(StringIO(scsv), delimiter=',', quotechar='"'):
                    cleaned_row = []
                    for value in row:
                        if value and value.startswith('"') and value.endswith('"'):
                            value = value.strip('"')
                        if value == "NULL":
                            value = None
                        cleaned_row.append(value)
                    params.append(cleaned_row)
            
                # Insert the data
                cols = params[0]; params = params[1:] 
                
                # Special handling for users table - encrypt passwords
                if table == 'users':
                    password_col_idx = cols.index('password')
                    for row in params:
                        if row[password_col_idx]:
                            row[password_col_idx] = self.onewayEncrypt(row[password_col_idx])
                
                self.insertRows(table=table, columns=cols, parameters=params)
                print(f"* Imported data for {table}")
            except FileNotFoundError:
                print(f"* No CSV file found for {table}")
            except Exception as e:
                print(f"* Error importing data for {table}: {e}")

        # Programmatically populate embeddings for initial data
        self.populate_initial_embeddings()

    def insertRows(self, table='table', columns=['x','y'], parameters=[['v11','v12'],['v21','v22']]):
        def process_value(value, query_params):
            if isinstance(value, str) and value.strip().startswith('(SELECT'):
                return value
            else:
                query_params.append(value)
                return '%s'
        
        has_multiple_rows = any(isinstance(el, list) for el in parameters)
        keys = ','.join(columns)
        query_params = []
        
        query = f"""INSERT INTO {table} ({keys}) VALUES """
        
        if has_multiple_rows:
            value_clauses = []
            for p in parameters:
                placeholders = [process_value(value, query_params) for value in p]
                value_clauses.append(f"({','.join(placeholders)})")
            query += ','.join(value_clauses)
        else:
            placeholders = [process_value(value, query_params) for value in parameters]
            query += f"""({','.join(placeholders)}) """                      
        
        query += " RETURNING *"

        result = self.query(query, query_params if query_params else None)
        if result and len(result) > 0:
            insert_id = result[0].get('id') or result[0].get(f'{table[:-1]}_id') or result[0].get(f'{table}_id')
        else:
            insert_id = None         
        return insert_id

    #--------------------------------------------------
    # AUTHENTICATION FUNCTIONS
    #--------------------------------------------------
    def authenticate(self, email='me@email.com', password='password'):
        check = self.query(query="""SELECT COUNT(*) as success FROM users WHERE email=%s AND password=%s""",
                           parameters=[email, self.onewayEncrypt(password)])[0]
        return check 

    def get_user_email(self, session_data=None):
        if session_data is None:
            from flask import session
            session_data = session
        if 'email' in session_data:
            return self.reversibleEncrypt('decrypt', session_data['email'])
        return 'Unknown'

    def get_user_role(self, session_data=None):
        if session_data is None:
            from flask import session
            session_data = session
        if 'email' in session_data:
            email = self.reversibleEncrypt('decrypt', session_data['email'])
            if email != 'Unknown':
                result = self.query("SELECT role FROM users WHERE email=%s", [email])
                if result:
                    return result[0]['role']
        return 'guest'

    #--------------------------------------------------
    # ENCRYPTION FUNCTIONS
    #--------------------------------------------------
    def onewayEncrypt(self, string):
        encrypted_string = hashlib.scrypt(string.encode('utf-8'),
                                          salt=self.encryption['oneway']['salt'],
                                          n=self.encryption['oneway']['n'],
                                          r=self.encryption['oneway']['r'],
                                          p=self.encryption['oneway']['p']
                                          ).hex()
        return encrypted_string

    def reversibleEncrypt(self, type, message):
        fernet = Fernet(self.encryption['reversible']['key'])
        if type == 'encrypt':
            message = fernet.encrypt(message.encode())
        elif type == 'decrypt':
            message = fernet.decrypt(message).decode()
        return message

    #--------------------------------------------------
    # PLANT DOCTOR & RAG VECTOR SEARCH
    #--------------------------------------------------
    def search_plant_treatment(self, query_text, query_embedding=None, limit=2):
        """
        البحث في المعرفة الزراعية (pgvector دلالياً مع مسار احتياطي نصي)
        """
        # 1. محاولة البحث الدلالي بالمتجهات إذا توفر التضمين
        if query_embedding:
            try:
                vector_sql = """
                    SELECT 
                        d.crop_name,
                        d.disease_name,
                        t.treatment_type,
                        t.active_ingredient,
                        t.trade_name,
                        t.dosage_per_liter,
                        t.safety_interval_days,
                        t.instructions
                    FROM plant_treatments t
                    JOIN plant_diseases d ON t.disease_id = d.disease_id
                    ORDER BY t.embedding <=> %s::vector
                    LIMIT %s;
                """
                results = self.query(vector_sql, (query_embedding, limit))
                if results:
                    return results
            except Exception as e:
                print(f"[Database] Vector search error: {e}")

        # 2. مسار احتياطي: بحث نصي مباشر إذا لم تكن المتجهات محسوبة
        try:
            fallback_sql = """
                SELECT 
                    d.crop_name,
                    d.disease_name,
                    t.treatment_type,
                    t.active_ingredient,
                    t.trade_name,
                    t.dosage_per_liter,
                    t.safety_interval_days,
                    t.instructions
                FROM plant_treatments t
                JOIN plant_diseases d ON t.disease_id = d.disease_id
                WHERE d.disease_name ILIKE %s OR d.symptoms ILIKE %s OR t.active_ingredient ILIKE %s
                LIMIT %s;
            """
            search_pattern = f"%{query_text}%"
            return self.query(fallback_sql, (search_pattern, search_pattern, search_pattern, limit))
        except Exception as e:
            print(f"[Database] Fallback search error: {e}")
            return []
# --------------------------------------------------
# Helper functions for Multi-Agent Database Experts
# --------------------------------------------------

def query_db(sql_query: str, parameters=None):
    """خبير القراءة: تنفيذ استعلامات SELECT واستعلامات البحث الدلالي pgvector"""
    db = database()
    return db.query(sql_query, parameters)

def modify_db(sql_query: str, parameters=None):
    """خبير الكتابة: تنفيذ التعديلات وتحديث التضمينات تلقائياً"""
    db = database()
    
    fix_sequences_sql = """
    DO $$
    DECLARE
        tbl text;
        col text;
        seq text;
    BEGIN
        FOR tbl, col IN VALUES ('skills', 'skill_id'), ('experiences', 'experience_id'), 
                               ('positions', 'position_id'), ('institutions', 'inst_id')
        LOOP
            seq := pg_get_serial_sequence(tbl, col);
            IF seq IS NOT NULL THEN
                EXECUTE format('SELECT setval(%L, COALESCE((SELECT MAX(%I) FROM %I), 0) + 1, false)', seq, col, tbl);
            END IF;
        END LOOP;
    END $$;
    """
    try:
        db.query(fix_sequences_sql)
    except Exception:
        pass
    
    result = db.query(sql_query, parameters)
    # تحديث التضمينات لأي صفوف جديدة مضافة
    db.populate_initial_embeddings()
    return result