# Copyright 2024 The Force-Directed Graph Embedding author.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import click

from .generate import cli_generate
from .embed import cli_embed
from .evaluate import cli_eval


@click.group()
def main():
    """Scripts to generate graphs, train and evaluate graph representations"""
    pass


main.add_command(cli_generate, "generate")
main.add_command(cli_embed, "embed")
main.add_command(cli_eval, "evaluate")

if __name__ == "__main__":
    main()