from pathlib import Path
import jaydebeapi
import os
from dotenv import load_dotenv

load_dotenv()

driver = "org.apache.hive.jdbc.HiveDriver"

env_url = os.getenv("INCORTA_ENV_URL")
tenant = os.getenv("INCORTA_TENANT")
sqlx_host = os.getenv("INCORTA_SQLX_HOST")
incorta_username = os.getenv("INCORTA_USERNAME")
PAT = os.getenv("PAT")

# jdbc_driver_path = f"{Path(__file__).parent}/hive-jdbc-2.3.8-standalone.jar"
jdbc_driver_path = "src/incorta/hive-jdbc-2.3.8-standalone.jar"

def connect_to_sqlx(driver, sqlx_host, incorta_username, tenant, PAT, jdbc_driver_path):
    """
    Connects to the SQLX interface of the Incorta environment.
    
    Args:
        driver: JDBC driver class name
        sqlx_host: The SQL interface host URL
        incorta_username: The Incorta username
        tenant: The tenant name
        PAT: Personal Access Token (JWT)
        jbdc_driver_path: Path to the JDBC driver JAR file
    
    Returns:
        Connection object on success, None on failure
    """
    try:
        return jaydebeapi.connect(
            driver,
            sqlx_host,
            [f"{incorta_username}%{tenant}", PAT],
            jdbc_driver_path
        )
    except Exception as e:
        print(f"JDBC connection error: {str(e)}")
        return None


def execute_query(spark_sql: str) -> dict:
    """
    Executes a Spark SQL query against the Incorta environment.

    Args:
        spark_sql: The Spark SQL query to execute

    Returns:
        dict: Contains either {"columns": [...], "rows": [...]} on success or {"error": "..."} on failure
    """
    spark_db_client = None
    cursor = None
    
    try:
        
        
        spark_sql = spark_sql.strip()
        
       
        try:
            spark_db_client = connect_to_sqlx(
                driver,
                sqlx_host,
                incorta_username,
                tenant,
                PAT,
                jdbc_driver_path
            )
        except Exception as e:
            return {"error": f"Failed to establish JDBC connection: {str(e)}"}
        
        if spark_db_client is None:
            return {
                "error": f"Failed to connect to Incorta SQL interface at {sqlx_host}. "
                        "Please verify the SQL interface URL and credentials."
            }
        
        try:
            cursor = spark_db_client.cursor()
        except Exception as e:
            return {"error": f"Failed to create database cursor: {str(e)}"}
        
        try:
            cursor.execute(spark_sql)
        except Exception as e:
            error_msg = str(e).lower()
            
            if "table" in error_msg and "not found" in error_msg:
                return {"error": f"Table or view not found. Please verify the table name in your query."}
            elif "column" in error_msg and "not found" in error_msg:
                return {"error": f"Column not found. Please verify the column names in your query."}
            elif "syntax" in error_msg or "parse" in error_msg:
                return {"error": f"SQL syntax error: {str(e)}"}
            elif "permission" in error_msg or "access denied" in error_msg:
                return {"error": f"Access denied. User does not have permission to execute this query."}
            elif "timeout" in error_msg:
                return {"error": "Query execution timed out. Please try a simpler query or add filters to reduce data volume."}
            else:
                return {"error": f"Query execution failed: {str(e)}"}
        
        try:
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        except Exception as e:
            return {"error": f"Failed to fetch query results: {str(e)}"}
        
        if not columns:
            return {"error": "Query returned no columns. This may indicate an issue with the query."}
        
        try:
            serializable_rows = []
            for row in rows:
                serializable_row = []
                for value in row:
                    if value is None:
                        serializable_row.append(None)
                    elif isinstance(value, (int, float, str, bool)):
                        serializable_row.append(value)
                    else:
                        serializable_row.append(str(value))
                serializable_rows.append(serializable_row)
        except Exception as e:
            return {"error": f"Failed to serialize query results: {str(e)}"}
        
        return {
            "columns": columns, 
            "rows": serializable_rows,
            "row_count": len(serializable_rows)
        }
    
    except Exception as e:
        return {"error": f"Unexpected error in execute_query: {str(e)}"}
    
    finally:
        try:
            if cursor:
                cursor.close()
        except Exception as e:
            print(f"Warning: Failed to close cursor: {str(e)}")
        
        try:
            if spark_db_client:
                spark_db_client.close()
        except Exception as e:
            print(f"Warning: Failed to close connection: {str(e)}")


if __name__ == "__main__":
    # Sample Spark SQL query
    sample_query = """
       SELECT 
    t.id AS ticket_id,
    t.subject,
    t.status,
    t.priority,
    t.type,
    t.created_at,
    t.updated_at,
    a.name AS assignee_name,
    r.name AS requester_name,
    o.name AS organization_name
    FROM ZendeskTickets.ticket t
    LEFT JOIN ZendeskTickets.assignee a ON t.assignee_id = a.id
    LEFT JOIN ZendeskTickets.requester r ON t.requester_id = r.id
    LEFT JOIN ZendeskTickets.organization o ON t.organization_id = o.id
    LIMIT 10
    """

    print("Executing sample Spark SQL query...")
    print(f"Query: {sample_query.strip()}")
    print("-" * 50)

    print(f"Found all credentials")
    print(f"   Environment URL: {env_url}")
    print(f"   Tenant: {tenant}")
    print(f"   Username: {incorta_username}")
    print(f"   SQLX Host: {sqlx_host[:50]}...")
    print(f"   PAT: {PAT[:50]}...")

    result = execute_query(sample_query)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Success! Retrieved {result['row_count']} rows")
        print(f"Columns: {result['columns']}")
        print("\nFirst few rows:")
        for i, row in enumerate(result['rows'][:5], 1):
            print(f"  Row {i}: {row}")