import subprocess

class GitTraining:
    def git_status(self):
        return subprocess.getoutput("git status")

    def git_commit(self, message):
        subprocess.run(["git", "commit", "-m", message])
        return "Commit successful"