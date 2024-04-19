"""
Prints settings as environment variables.
"""

if __name__ == "__main__":
    # if this script is run, then it exports the configuration as environment
    # variables (for bash/R, etc)

    from proj.conf import env_vars

    for k, v in env_vars.items():
        print(f"export {k}={v}")
