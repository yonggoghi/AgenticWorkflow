#!/usr/bin/env python3
"""
Test script for database configuration and connection.
"""
import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Also add parent directory to handle package imports
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded successfully")
except ImportError:
    print("❌ dotenv not available")

# Import configuration
try:
    from config.settings import DATABASE_CONFIG, DATA_CONFIG
except ImportError:
    from mms_extractor.config.settings import DATABASE_CONFIG, DATA_CONFIG

def test_database_config():
    """Test database configuration."""
    print("\n=== Database Configuration Test ===")
    print(f"📊 Data source: {DATA_CONFIG.offer_info_data_src}")
    print(f"🏠 DB Host: {DATABASE_CONFIG.db_host or 'Not set'}")
    print(f"👤 DB Username: {DATABASE_CONFIG.db_username or 'Not set'}")
    print(f"🔐 DB Password: {'***' if DATABASE_CONFIG.db_password else 'Not set'}")
    print(f"🚪 DB Port: {DATABASE_CONFIG.db_port}")
    print(f"🗄️ DB Service Name: {DATABASE_CONFIG.db_service_name or 'Not set'}")
    
    # Check if configuration is valid
    is_valid = DATABASE_CONFIG.validate()
    print(f"✅ Configuration valid: {is_valid}")
    
    if not is_valid:
        print("\n❌ Missing required environment variables:")
        if not DATABASE_CONFIG.db_username:
            print("  - DB_USERNAME")
        if not DATABASE_CONFIG.db_password:
            print("  - DB_PASSWORD")
        if not DATABASE_CONFIG.db_host:
            print("  - DB_HOST")
        if not DATABASE_CONFIG.db_service_name:
            print("  - DB_NAME")
    
    return is_valid

def test_database_connection():
    """Test actual database connection."""
    print("\n=== Database Connection Test ===")
    
    if not DATABASE_CONFIG.validate():
        print("❌ Cannot test connection: Invalid configuration")
        return False
    
    try:
        import cx_Oracle
        print("✅ cx_Oracle imported successfully")
    except ImportError:
        print("❌ cx_Oracle not available. Install with: pip install cx_Oracle")
        return False
    
    try:
        # Create DSN
        dsn = cx_Oracle.makedsn(
            DATABASE_CONFIG.db_host, 
            DATABASE_CONFIG.db_port, 
            service_name=DATABASE_CONFIG.db_service_name
        )
        print(f"✅ DSN created: {dsn}")
        
        # Test connection
        conn = cx_Oracle.connect(
            user=DATABASE_CONFIG.db_username,
            password=DATABASE_CONFIG.db_password,
            dsn=dsn,
            encoding="UTF-8"
        )
        print("✅ Database connection successful!")
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM DUAL")
        result = cursor.fetchone()
        print(f"✅ Test query successful: {result}")
        
        cursor.close()
        conn.close()
        print("✅ Connection closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_data_loading():
    """Test data loading with database."""
    print("\n=== Data Loading Test ===")
    
    # Temporarily set data source to db
    original_src = DATA_CONFIG.offer_info_data_src
    DATA_CONFIG.offer_info_data_src = "db"
    
    try:
        # Import DataManager with fallback strategies
        try:
            from core.data_manager import DataManager
        except ImportError:
            from mms_extractor.core.data_manager import DataManager
        
        print("🔄 Creating DataManager...")
        data_manager = DataManager(use_mock_data=False)
        
        print("🔄 Loading item data from database...")
        item_df = data_manager.load_item_data()
        
        print(f"✅ Loaded {len(item_df)} items from database")
        print(f"📋 Columns: {list(item_df.columns)}")
        
        # Safe data display with error handling
        print(f"🔍 Sample data (first 3 rows):")
        try:
            # Try simple string representation first
            sample_data = item_df.head(3)
            for idx, (i, row) in enumerate(sample_data.iterrows()):
                print(f"  Row {idx + 1}: item_nm='{row.get('item_nm', 'N/A')}', item_id='{row.get('item_id', 'N/A')}', domain='{row.get('domain', 'N/A')}'")
                if idx >= 2:  # Limit to 3 rows
                    break
        except Exception as display_error:
            print(f"  ⚠️ Display error (non-critical): {display_error}")
            print(f"  📊 Data shape: {item_df.shape}")
        
        # Additional validation
        if len(item_df) > 0:
            print(f"✅ Data loading test PASSED - {len(item_df)} records loaded successfully")
            return True
        else:
            print(f"❌ Data loading test FAILED - No records loaded")
            return False
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        # Print more detailed error information
        import traceback
        print(f"📋 Detailed error:")
        traceback.print_exc()
        return False
    finally:
        # Restore original data source
        DATA_CONFIG.offer_info_data_src = original_src

if __name__ == "__main__":
    print("🧪 MMS Extractor Database Configuration Test")
    print("=" * 50)
    
    # Test configuration
    config_ok = test_database_config()
    
    if config_ok:
        # Test connection
        conn_ok = test_database_connection()
        
        if conn_ok:
            # Test data loading
            data_ok = test_data_loading()
            
            if data_ok:
                print("\n🎉 All tests passed!")
            else:
                print("\n❌ Data loading test failed")
        else:
            print("\n❌ Connection test failed")
    else:
        print("\n❌ Configuration test failed")
        print("\n💡 To set up database connection:")
        print("1. Create a .env file in the mms_extractor directory")
        print("2. Add the following variables:")
        print("   OFFER_INFO_DATA_SRC=db")
        print("   DB_USERNAME=your_username")
        print("   DB_PASSWORD=your_password") 
        print("   DB_HOST=your_host")
        print("   DB_PORT=1521")
        print("   DB_NAME=your_service_name") 