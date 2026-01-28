# Zeppelin server address
ZEPP_URL = "http://150.6.14.94:30132"
NOTEBOOK_ID = "2MGAYX8RV"

# Paragraph IDs (Pre)
PARAGRAPH_IDS_PRE = [
    "paragraph_1764658338256_686533166",
    "paragraph_1764742922351_426209997",
    "paragraph_1764742953919_436300403",
    "paragraph_1764659911196_1763551717",
    "paragraph_1764641394585_598529380",
    "paragraph_1764739202982_181479704",
    "paragraph_1764739017819_1458690185",
    "paragraph_1764738582669_1614068999",
    "paragraph_1764756027560_85739584",
]

# Paragraph IDs (Main)
PARAGRAPH_IDS = [
    "paragraph_1766323923540_1041552789",
    "paragraph_1767594403472_2124174124",
    "paragraph_1764755002817_1620624445",
    "paragraph_1764832142136_413314670",
    "paragraph_1766224516076_433149416",
]

# Parameters - Nested Loop Support
# 상위 루프: sendMonth (먼저 실행)
PARAMS_OUTER = [f"sendMonth:{ym}" for ym in range(202510, 202513)]  # 또는 ["sendMonth:202511", "sendMonth:202512"]

# 하위 루프: suffix (각 sendMonth마다 반복)
PARAMS_INNER = [f"suffix:{hex(i)[2:]}" for i in range(0, 16)]  # 0-f 전체

# PARAMS_OUTER x PARAMS_INNER = 1 x 16 = 16번 실행
# 예: sendMonth=202512, suffix=0
#     sendMonth=202512, suffix=1
#     ...
#     sendMonth=202512, suffix=f
# 
# PARAMS_OUTER를 ["sendMonth:202511", "sendMonth:202512"]로 설정하면:
# 2 x 16 = 32번 실행

# PARAMS는 자동 생성됨 (PARAMS_OUTER x PARAMS_INNER)
# 직접 지정하려면 아래 주석 해제:
# PARAMS = [
#     [f"sendMonth:{month}", f"suffix:{hex(i)[2:]}"]
#     for month in ["202511", "202512"]
#     for i in range(16)
# ]

# Spark restart options
RESTART_SPARK_AT_START = True   # Restart Spark before starting PRE paragraphs
RESTART_SPARK_AT_END = True     # Restart Spark after all paragraphs complete

# Smart Detection: Output check function (optional)
def is_param_completed(param_list):
    """
    Check if the output for given parameter combination already exists.
    This enables smart detection to skip already completed work.
    
    Args:
        param_list: List of "key:value" strings
                   Example: ["sendMonth:202512", "suffix:c"]
    
    Returns:
        bool: True if output exists (skip execution), False otherwise
    
    Usage:
        - Return True to skip execution (output already exists)
        - Return False to execute (output missing or incomplete)
        - Raise exception to execute with warning
    """
    import subprocess
    
    # Extract parameter values
    params_dict = {}
    for param in param_list:
        key, value = param.split(":", 1)
        params_dict[key] = value
    
    sendMonth = params_dict.get("sendMonth")
    suffix = params_dict.get("suffix")
    
    if not sendMonth or not suffix:
        return False  # Missing parameters, execute
    
    # Check output path (match Scala output path)
    # Adjust this path according to your Scala code's output location
    output_paths = [
        f"aos/sto/trainDFRev10/send_ym={sendMonth}/suffix={suffix}",
        f"aos/sto/testDFRev/send_ym={sendMonth}/suffix={suffix}",
    ]
    
    # Check if all required output paths exist
    try:
        for path in output_paths:
            result = subprocess.run(
                ["hdfs", "dfs", "-test", "-e", path],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                # Path doesn't exist
                return False
        
        # All paths exist
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Hadoop command not available or timed out, execute to be safe
        return False
    except Exception as e:
        # Other errors, log and execute
        print(f"  Warning: Output check failed ({e}), will execute")
        return False