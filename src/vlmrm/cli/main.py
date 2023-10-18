import typer
from vlmrm.cli.generate_dataset import generate_dataset
from vlmrm.cli.train import train

app = typer.Typer()

app.command("train")(train)
app.command("generate_dataset")(generate_dataset)


def main():
    app()


if __name__ == "__main__":
    app()
