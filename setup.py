"""Package the shared runtime, but not the executable research experiments beside it."""
from setuptools import setup
from setuptools.command.build_py import build_py


class RuntimeBuild(build_py):
    runtime_modules = {"__init__", "ane", "mil", "layers", "delta", "lutq", "runtime", "service", "ref_backbone", "errors"}

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == "clef_ane.engine":
            modules = [module for module in modules if module[1] in self.runtime_modules]
        return modules


setup(cmdclass={"build_py": RuntimeBuild})
