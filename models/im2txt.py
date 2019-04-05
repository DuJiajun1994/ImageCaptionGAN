from models.text_embedding import TextEmbedding


class Im2Txt(TextEmbedding):
    def __init__(self, args):
        super(Im2Txt, self).__init__(fc_feat_size=args.fc_feat_size, att_feat_size=args.att_feat_size, args=args)
