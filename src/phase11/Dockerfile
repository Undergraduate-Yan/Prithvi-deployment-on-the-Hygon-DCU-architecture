FROM harbor.sourcefind.cn:5443/dcu/admin/base/pytorch@sha256:52f2f4f7d4cc3e420b379f9212e1983d52fd3fd31e7427edcf8a07bf62acc069

ARG MIGRAPHX_RUN=migraphx-5.1.0+das.opt1.dtk25042-cp310-cp310-manylinux_2_28_x86_64.run
COPY ${MIGRAPHX_RUN} /tmp/migraphx.run
COPY wheels/*.whl /tmp/wheels/

# The vendor installer asks for an installation prefix and whether to install
# the optional diffusers package.  Install only the core runtime into this
# image's DTK 25.04.2 tree; never touch the host DTK installation.
RUN cd /tmp \
    && printf '/opt/dtk\nn\n' | bash /tmp/migraphx.run \
    && rm -f /tmp/migraphx.run \
    && rm -rf /tmp/migraphx_run_installer_temp \
    && /opt/dtk/bin/migraphx-driver --version

# ONNX is used only to generate and inspect the frozen synthetic Phase 7-10
# graphs. Install locked local wheels and keep the vendor ONNX Runtime supplied
# by the matched base image.
RUN python -m pip install --no-cache-dir --no-index --find-links=/tmp/wheels \
        'ml_dtypes==0.5.4' 'onnx==1.22.0' \
    && rm -rf /tmp/wheels

ENV PYTHONPATH=/opt/dtk/lib:${PYTHONPATH} \
    LD_PRELOAD=/opt/dtk/lib/libmigraphx_gpu.so
LABEL org.opencontainers.image.title="K100 Phase7-10 MIGraphX matched environment" \
      org.opencontainers.image.base.digest="sha256:52f2f4f7d4cc3e420b379f9212e1983d52fd3fd31e7427edcf8a07bf62acc069" \
      dtk.version="25.04.2" \
      migraphx.version="5.1.0+das.opt1.dtk25042" \
      migraphx.package.sha256="5039e590de56a7e5da7fcfbe6acda1150d02140cc0e47a5f04c97cf41f980074"
