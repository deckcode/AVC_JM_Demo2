import numpy as  np

class CABACEncoder:
    def __init__(self):
        # 初始化上下文模型 (JM 9.3.4.3)
        self.contexts = [{'mps':0, 'state':0} for _ in range(398)]
        self.range = 510
        self.low = 0
        self.bits_left = 23
        self.ff_byte = 0xFF
        self.ff_count = 0
        self.bitstream = bytearray()

        # 预计算概率表 (JM 9.3.4.4表9-41)
        self.lps_range = [
            6798, 7476, 8214, 8910, 9532, 10082, 10554, 10974,
            11338, 11650, 11918, 12150, 12350, 12522, 12670, 12798
        ]
        self.transit_table = [
            [0,1], [2,3], [4,5], [6,7],
            [8,9], [10,11], [12,13], [14,15],
            [16,17], [18,19], [20,21], [22,23],
            [24,25], [26,27], [28,29], [30,31]
        ]

    def encode_block(self, coeffs, component='luma', blk_type='inter'):
        """ 编码4x4残差块 (JM 9.3.4) """
        # 二进制化流程
        bin_str = self._binarize(coeffs, blk_type)

        # 初始化算术编码引擎
        self._init_encoder()

        # 逐位编码
        for bit, ctx_idx in bin_str:
            self._encode_bit(bit, ctx_idx)

        # 终止编码
        self._terminate()
        return bytes(self.bitstream)

    def _binarize(self, coeffs, blk_type):
        """ 二进制化流程 (JM 9.3.4.2) """
        # 扫描顺序选择
        scan = self._get_scan_order(blk_type)
        scanned = coeffs.flatten()[scan]


        #  bins列表可能用于存储二进制化的符号，也就是编码过程中的二进制位流。
        #  coeff_list可能用来存储非零系数的列表，或者记录需要处理的系数信息。
        #  last_pos初始化为-1，可能用于跟踪最后一个非零系数的位置，初始值-1可能表示尚未找到任何非零系数。

        # 1. 编码significant_coeff_flag
        bins = []
        coeff_list = []
        last_pos = -1

        for i, c in enumerate(scanned):
            if c != 0:
                ctx_idx = self._get_sig_ctx(i, scan)
                bins.append((1, ctx_idx))
                coeff_list.append((i, c))
                last_pos = i
            else:
                ctx_idx = self._get_sig_ctx(i, scan)
                bins.append((0, ctx_idx))

        # 2. 编码last_significant_coeff_flag
        # 该代码遍历系数列表，为每个元素生成二进制位与上下文索引。
        # 通过判断当前元素是否是列表最后一个，决定二进制位的值（1或0），并将结果与通过位置和扫描顺序计算的上下文索引一起存入列表。
        for idx, (pos, _) in enumerate(coeff_list):
            ctx_idx = self._get_last_ctx(pos, scan)
            bins.append((1 if idx == len(coeff_list)-1 else 0, ctx_idx))

        # 3. 编码coeff_abs_level_minus1
        for pos, coeff in reversed(coeff_list):
            abs_level = abs(coeff) - 1
            prefix = abs_level // 15
            suffix = abs_level % 15

            # 前缀编码
            ctx_idx_base = 16 + min(4, pos//4)
            for _ in range(prefix):
                bins.append((1, ctx_idx_base))
            bins.append((0, ctx_idx_base))

            # 后缀编码 (4位定长)
            if suffix >= 0:
                bins += [( (suffix>>3)&1, 21 ), ( (suffix>>2)&1, 22 ),
                         ( (suffix>>1)&1, 23 ), ( suffix&1, 24 )]

            # 符号位
            bins.append((0 if coeff > 0 else 1, 25))

        return bins

    def _get_scan_order(self, blk_type):
        """ 获取扫描顺序 (JM 9.3.4.1) """
        return [
            0, 1, 4, 8,   # 之字形扫描
            5, 2, 3, 6,
            9, 12, 13, 10,
            7, 11, 14, 15
        ] if blk_type == 'inter' else [
            0, 4, 1, 8,   # 场扫描
            12, 5, 9, 2,
            3, 6, 10, 13,
            7, 11, 14, 15
        ]

    # 该函数计算视频编码中4x4块内某位置相邻系数的上下文值。将pos转为(x,y)坐标后，检查左侧和上方的扫描系数是否非零，若存在非零则标记为1，
    # 最终返回left+top的最小值（不超过5）。
    def _get_sig_ctx(self, pos, scan):
        """ 计算significant_coeff_flag上下文 (JM 9.3.4.3) """
        x, y = pos%4, pos//4
        left = 1 if x>0 and scan[pos-1] != 0 else 0
        top = 1 if y>0 and scan[pos-4] != 0 else 0
        return min(5, left + top)

    def _get_last_ctx(self, pos, scan):
        """ 计算last_significant_coeff_flag上下文 """
        return 6 + min(2, (15 - pos) // 4)

    def _encode_bit(self, bit, ctx_idx):
        """ 算术编码核心 (JM 9.3.4.4) """
        ctx = self.contexts[ctx_idx]
        p_state = ctx['state']
        val_mps = ctx['mps']

        # 计算LPS区间
        q_range = (self.range * self.lps_range[p_state]) >> 16
        self.range -= q_range

        # 更新区间和低位
        if bit != val_mps:
            self.low += self.range
            self.range = q_range
            if p_state == 0:
                ctx['mps'] = 1 - val_mps
            ctx['state'] = self.transit_table[p_state][1]
        else:
            ctx['state'] = self.transit_table[p_state][0]

        # 重新归一化
        while self.range < 256:
            self.range <<= 1
            self.low <<= 1
            self.bits_left -= 1

            if self.bits_left < 0:
                self._write_byte()

    def _write_byte(self):
        """ 字节输出处理 (JM 9.3.4.5) """
        byte = (self.low >> 23) & 0xFF
        self.low &= 0x7FFFFF

        if byte == 0xFF:
            self.ff_count += 1
        else:
            if self.ff_byte != 0xFF:
                self.bitstream.append(self.ff_byte)
                for _ in range(self.ff_count):
                    self.bitstream.append(0xFF)
                self.ff_count = 0
            self.ff_byte = byte

        self.bits_left += 8

    def _terminate(self):
        """ 终止编码流程 (JM 9.3.4.5) """
        # 输出剩余字节
        self.range -= 2
        self.low += self.range
        self.range = 2

        for _ in range(2):
            self.range <<= 1
            self.low <<= 1
            self.bits_left -= 1
            if self.bits_left < 0:
                self._write_byte()

        # 写入最后字节
        if self.ff_byte != 0xFF:
            self.bitstream.append(self.ff_byte)
        for _ in range(self.ff_count):
            self.bitstream.append(0xFF)

    def _init_encoder(self):
        """ 初始化算术编码器状态 """
        self.range = 510
        self.low = 0
        self.bits_left = 23
        self.ff_byte = 0xFF
        self.ff_count = 0
        self.bitstream = bytearray()



# 测试用例
if __name__ == "__main__":
    # 测试块来自JM软件测试向量
    test_block = np.array([
        [3, -1, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    encoder = CABACEncoder()
    bitstream = encoder.encode_block(test_block, 'luma', 'inter')

    print(f"编码结果 (Hex): {bitstream.hex().upper()}")
    print(f"编码字节数: {len(bitstream)}")
    print("预期输出 (JM参考): 0x9B9050")
