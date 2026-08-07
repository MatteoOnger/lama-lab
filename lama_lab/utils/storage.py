import dill
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


class ExperimentManager:
    """Represents a single experiment directory for managing artifacts.

    Provides high-level methods to save and load data artifacts such as
    JSON files, PyTorch tensors, matplotlib figures, text files, and pickled
    objects within a dedicated experiment directory.

    Parameters
    ----------
    path : str or Path
        Path to the experiment directory. The directory will be created if it
        does not exist.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

        self._save_dispatch = {
            torch.Tensor: self.save_tensor,
            Figure: self.save_figure,
            str: self.save_text,
        }

        self._load_dispatch = {
            ".pt": self.load_tensor,
            ".json": self.load_json,
            ".pkl": self.load_pickle,
            ".txt": self.load_text,
        }
        return

    def _get_path(self, name: str | Path, extension: str) -> Path:
        """Resolve a file path within the experiment directory and append the extension if missing.

        Parameters
        ----------
        name : str or Path
            Base file name or relative path.
        extension : str
            Expected file extension, including the leading dot (e.g. ``".json"``).

        Returns
        -------
        file_path : Path
            Full path resolved within the experiment directory, guaranteed to end
            with the specified extension.
        """
        file_path = self.file(name)
        if not file_path.name.endswith(extension):
            file_path = file_path.with_name(f"{file_path.name}{extension}")
        return file_path

    def file(self, filename: str | Path) -> Path:
        """Construct a full path for a file inside the experiment directory.

        Parameters
        ----------
        filename : str or Path
            Name or relative path of the file.

        Returns
        -------
        path : Path
            Full path resolved within the experiment directory.
        """
        return self.path / filename

    def load_all(self) -> dict[str, Any]:
        """Load all recognized files from the experiment directory into a dictionary.

        Files are loaded automatically based on their extension. Unrecognized
        extensions are silently ignored.

        Returns
        -------
        artifacts : dict of str to Any
            Dictionary mapping full file names (including extensions) to their
            loaded Python objects.
        """
        artifacts = {}
        for file_path in self.path.iterdir():
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            if ext in self._load_dispatch:
                load_func = self._load_dispatch[ext]
                artifacts[file_path.name] = load_func(file_path.name)
        return artifacts

    def load_json(self, name: str | Path, encoding: str = "utf-8") -> Any:
        """Load data from a JSON file.

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        encoding : str, optional
            Text encoding, by default ``"utf-8"``.

        Returns
        -------
        artifact : Any
            Deserialized JSON object.

        Raises
        ------
        FileNotFoundError
            If the requested JSON file does not exist.
        """
        file_path = self._get_path(name, ".json")
        if not file_path.is_file():
            raise FileNotFoundError(f"JSON file not found: {file_path}.")
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)

    def load_pickle(self, name: str | Path) -> Any:
        """Load and deserialize an object from disk using dill.

        Parameters
        ----------
        name : str or Path
            File name or relative path.

        Returns
        -------
        obj : Any
            Deserialized Python object.

        Raises
        ------
        FileNotFoundError
            If the requested pickle file does not exist.
        """
        file_path = self._get_path(name, ".pkl")
        if not file_path.is_file():
            raise FileNotFoundError(f"Pickle file not found: {file_path}.")
        with open(file_path, "rb") as f:
            return dill.load(f)

    def load_tensor(
        self,
        name: str | Path,
        map_location: torch.device | str | None = None,
        weights_only: bool = True,
    ) -> torch.Tensor:
        """Load a PyTorch tensor from disk.

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        map_location : torch.device, str, or None, optional
            Device location for loading the tensor, by default None.
        weights_only : bool, optional
            Whether to restrict unpickling to tensors and primitive types for
            security, by default True.

        Returns
        -------
        tensor : torch.Tensor
            Loaded PyTorch tensor.

        Raises
        ------
        FileNotFoundError
            If the requested tensor file does not exist.
        """
        file_path = self._get_path(name, ".pt")
        if not file_path.is_file():
            raise FileNotFoundError(f"Tensor file not found: {file_path}.")
        return torch.load(
            file_path, map_location=map_location, weights_only=weights_only
        )

    def load_text(self, name: str | Path, encoding: str = "utf-8") -> str:
        """Load text content from a file.

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        encoding : str, optional
            Text encoding, by default ``"utf-8"``.

        Returns
        -------
        text : str
            Text file content.

        Raises
        ------
        FileNotFoundError
            If the requested text file does not exist.
        """
        file_path = self._get_path(name, ".txt")
        if not file_path.is_file():
            raise FileNotFoundError(f"Text file not found: {file_path}.")
        return file_path.read_text(encoding=encoding)

    def save_all(self, artifacts: dict[str, Any]) -> None:
        """Save a dictionary of heterogeneous objects automatically routing them to the correct format.

        Parameters
        ----------
        artifacts : dict of str to Any
            Dictionary where keys are file names (or base names) and values are
            the objects to save. The correct saving method is automatically inferred
            from the object's type.
        """
        for name, obj in artifacts.items():
            saved = False

            for obj_type, save_func in self._save_dispatch.items():
                if isinstance(obj, obj_type):
                    save_func(name, obj)
                    saved = True
                    break

            if not saved:
                try:
                    json.dumps(obj)
                    self.save_json(name, obj)
                except TypeError:
                    self.save_pickle(name, obj)
        return

    def save_figure(
        self,
        name: str | Path,
        fig: Figure | None = None,
        dpi: int = 300,
        fmt: str = "png",
        close: bool = False,
    ) -> Path:
        """Save a Matplotlib figure to disk.

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        fig : matplotlib.figure.Figure, optional
            Figure object to save. If ``None``, uses current active figure.
        dpi : int, optional
            Resolution in dots per inch, by default 300.
        fmt : str, optional
            Image format extension (e.g. ``"png"``, ``"pdf"``), by default ``"png"``.
        close : bool, optional
            Whether to close the figure after saving to free memory, by default False.

        Returns
        -------
        path : Path
            Path to the saved figure file.
        """
        extension = f".{fmt.lstrip('.')}"
        file_path = self._get_path(name, extension)

        if fig is None:
            fig = plt.gcf()
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")

        if close:
            plt.close(fig)
        return file_path

    def save_json(
        self, name: str | Path, obj: Any, indent: int = 4, encoding: str = "utf-8"
    ) -> Path:
        """Save an object as a JSON file.

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        obj : Any
            JSON-serializable object.
        indent : int, optional
            Indentation level for formatting, by default 4.
        encoding : str, optional
            Text encoding, by default ``"utf-8"``.

        Returns
        -------
        path : Path
            Path to the saved JSON file.
        """
        file_path = self._get_path(name, ".json")
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(obj, f, indent=indent)
        return file_path

    def save_pickle(self, name: str | Path, obj: Any) -> Path:
        """Serialize and save an object using dill (extended pickle).

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        obj : Any
            Object to serialize. Supports complex types like lambdas.

        Returns
        -------
        path : Path
            Path to the saved pickle file.
        """
        file_path = self._get_path(name, ".pkl")
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        return file_path

    def save_tensor(self, name: str | Path, tensor: torch.Tensor) -> Path:
        """Save a PyTorch tensor to disk.

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        tensor : torch.Tensor
            PyTorch tensor to save.

        Returns
        -------
        path : Path
            Path to the saved tensor file.
        """
        file_path = self._get_path(name, ".pt")
        torch.save(tensor, file_path)
        return file_path

    def save_text(self, name: str | Path, text: str, encoding: str = "utf-8") -> Path:
        """Save text content to a file.

        Parameters
        ----------
        name : str or Path
            File name or relative path.
        text : str
            Text string to write.
        encoding : str, optional
            Text encoding, by default ``"utf-8"``.

        Returns
        -------
        path : Path
            Path to the saved text file.
        """
        file_path = self._get_path(name, ".txt")
        file_path.write_text(text, encoding=encoding)
        return file_path

    def __enter__(self) -> "ExperimentManager":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Exit the context manager."""
        return False


class ResultsManager:
    """Manager for creating and organizing experiment directories.

    Parameters
    ----------
    root : str or Path
        Root directory where all experiment subdirectories will be stored.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        return

    def get_experiment(self, name: str | Path) -> ExperimentManager:
        """Load an existing experiment directory.

        Parameters
        ----------
        name : str or Path
            Directory name or relative path under the root directory.

        Returns
        -------
        exp : Experiment
            An initialized ``Experiment`` object bound to the existing directory.

        Raises
        ------
        FileNotFoundError
            If the experiment directory does not exist.
        """
        path = self.root / name
        if not path.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {path}")
        return ExperimentManager(path)

    def list_experiments(self) -> list[Path]:
        """List all experiment subdirectories under root sorted by creation time.

        Returns
        -------
        exps : list of Path
            List of experiment directory paths.
        """
        return sorted([p for p in self.root.iterdir() if p.is_dir()])

    def new_experiment(self, name: str | None = None) -> ExperimentManager:
        """Create a new timestamped experiment directory.

        Parameters
        ----------
        name : str, optional
            Optional descriptive suffix for the experiment folder name.

        Returns
        -------
        exp : Experiment
            An initialized ``Experiment`` object bound to the created directory.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dirname = timestamp if name is None else f"{timestamp}_{name}"
        path = self.root / dirname

        i = 1
        while path.exists():
            path = self.root / f"{dirname}_{i}"
            i += 1

        path.mkdir(parents=True, exist_ok=False)
        return ExperimentManager(path)
