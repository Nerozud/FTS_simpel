# A dictionary with "enable_grouping" set to True
env_config = {"enable_grouping": True}

if env_config.get("enable_grouping", False):
    print("Grouping is enabled.")
# else:
#     print("Grouping is not enabled.")
