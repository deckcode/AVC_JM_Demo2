from

class CABACDecoder:
    def __init__(self, bitstream):
        self.bitstream = bitstream
        self.ctx_models = [{'mps':0, 'state':0} for _ in range(398)]
        self.range = 510
        self.code = 0
        self.pos = 0
        self.ff_count = 0
        self._init_decoder()

    def _init_decoder(self):
        """ 初始化算术解码器 (JM 9.3.4.5) """
        self.code = (self.bitstream[0] << 24) | (self.bitstream[1] << 16)
        self.pos = 2
        self.range = 510

        # 跳过起始字节
        if self.bitstream[0] == 0xFF:
            self.pos += 1

    def decode_block(self, component='luma', blk_type='inter'):
        """ 解码4x4残差块 """
        # 初始化空系数矩阵
        coeffs = np.zeros((4,4), dtype=np.int16)

        # 二进制化解码流程
        bin_list = []

        # 1. 解码significant_coeff_flag
        sig_coeffs = []
        scan = self._get_scan_order(blk_type)
        for i in range(16):
            ctx_idx = self._get_sig_ctx(i, scan)
            bit = self._decode_bit(ctx_idx)
            if bit:
                sig_coeffs.append(i)

        # 2. 解码last_significant_coeff_flag
        last_pos = -1
        for i in range(len(sig_coeffs)):
            ctx_idx = self._get_last_ctx(sig_coeffs[i], scan)
            if self._decode_bit(ctx_idx):
                last_pos = sig_coeffs[i]
                break

        # 3. 解码coeff_abs_level_minus1
        levels = []
        for pos in reversed(sig_coeffs[:sig_coeffs.index(last_pos)+1]):
            # 解码前缀
            prefix = 0
            ctx_idx = 16 + min(4, pos//4)
            while self._decode_bit(ctx_idx):
                prefix += 1

            # 解码后缀
            suffix = 0
            if prefix > 0:
                for i in range(4):
                    ctx_idx = 21 + i
                    suffix = (suffix << 1) | self._decode_bit(ctx_idx)

            abs_level = prefix * 15 + suffix
            sign = self._decode_bit(25)
            levels.append(abs_level + 1 if sign == 0 else -(abs_level + 1))

        # 重构系数矩阵
        scan_order = self._get_scan_order(blk_type)
        for pos, level in zip(reversed(sig_coeffs), levels):
            x, y = pos % 4, pos // 4
            coeffs[y, x] = level

        return coeffs

    def _decode_bit(self, ctx_idx):
        """ 算术解码核心 (JM 9.3.4.4) """
        ctx = self.ctx_models[ctx_idx]
        p_state = ctx['state']
        val_mps = ctx['mps']

        # 计算LPS区间
        q_range = (self.range * CABACEncoder.lps_range[p_state]) >> 16
        lps_range = q_range
        mps_range = self.range - lps_range

        # 判断符号
        bit = val_mps
        if self.code >= mps_range:
            bit = 1 - val_mps
            self.code -= mps_range
            self.range = lps_range
            if p_state == 0:
                ctx['mps'] = 1 - val_mps
            ctx['state'] = CABACEncoder.transit_table[p_state][1]
        else:
            self.range = mps_range
            ctx['state'] = CABACEncoder.transit_table[p_state][0]

        # 重新归一化
        while self.range < 256:
            self.range <<= 1
            self.code = (self.code << 1) & 0xFFFFFFFF
            if self.pos < len(self.bitstream):
                self.code |= (self.bitstream[self.pos] >> 7)
                if (self.bitstream[self.pos] & 0xFF) == 0xFF:
                    self.pos += 1
            self.pos += 1

        return bit

    # 复用编码器的扫描和上下文计算
    _get_scan_order = CABACEncoder._get_scan_order
    _get_sig_ctx = CABACEncoder._get_sig_ctx
    _get_last_ctx = CABACEncoder._get_last_ctx

# 测试用例
if __name__ == "__main__":
    # 原始测试块
    original = np.array([
        [3, -1, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # 编码
    encoder = CABACEncoder()
    bitstream = encoder.encode_block(original, 'luma', 'inter')

    # 解码
    decoder = CABACDecoder(bitstream)
    decoded = decoder.decode_block('luma', 'inter')

    print("原始系数矩阵:")
    print(original)
    print("\n解码系数矩阵:")
    print(decoded)
    print("\n解码结果验证:", np.array_equal(original, decoded))
