# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.fused_moe.core import _validate_routing_replay_out


def test_routing_replay_buffer_must_cover_active_tokens():
    _validate_routing_replay_out(torch.empty((8, 2), dtype=torch.int16), 2, 8)
    _validate_routing_replay_out(torch.empty((16, 2), dtype=torch.int16), 2, 8)

    with pytest.raises(ValueError, match="cover every active token"):
        _validate_routing_replay_out(torch.empty((7, 2), dtype=torch.int16), 2, 8)
