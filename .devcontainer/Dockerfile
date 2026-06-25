FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ARG USERNAME=vscode

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    bash-completion \
    jq \
    yq \
    && rm -rf /var/lib/apt/lists/*


# COMMAND HISTORY
RUN mkdir -p /commandhistory && \
    touch /commandhistory/.bash_history && \
    chown -R ${USERNAME}:${USERNAME} /commandhistory

RUN echo "export PROMPT_COMMAND='history -a'" >> /home/${USERNAME}/.bashrc && \
    echo "export HISTFILE=/commandhistory/.bash_history" >> /home/${USERNAME}/.bashrc && \
    echo "export HISTSIZE=100000" >> /home/${USERNAME}/.bashrc && \
    echo "export HISTFILESIZE=100000" >> /home/${USERNAME}/.bashrc && \
    echo "source /usr/share/bash-completion/bash_completion" >> /home/${USERNAME}/.bashrc

USER ${USERNAME}

# UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/home/${USERNAME}/.venv
