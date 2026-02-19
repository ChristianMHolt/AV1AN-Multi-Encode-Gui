
import sys
import subprocess

def main():
    real_exe = "/app/svt"
    args = sys.argv[1:]
    new_args = []
    i = 0
    while i < len(args):
        if args[i] == '-b':
            new_args.append('--output')
        else:
            new_args.append(args[i])
        i += 1

    # Run the real executable
    try:
        sys.exit(subprocess.call([real_exe] + new_args))
    except Exception as e:
        print(f"Shim Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
