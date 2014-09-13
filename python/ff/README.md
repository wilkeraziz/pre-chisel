# Feature Functions

This framework is based on cdec's.
To add a new feature function, simply implement your idea using this very simple guidelines:

1. Import from `ff`

        import ff

2. Configure your extractor

        @ff.configure
        def thisMethodConfiguresMyNewFeature(config):
            """
            config is a dictionary containing the strings parsed from chisel's config.ini
            here you can load stuff into memory (e.g. pre-trained models)
            """
            pass

3. Implement your features

        @ff.dense
        def MyFeature(hypothesis):
            """
            hypothesis contains the input and the translation
            this function must return 1 real value
            in this case the feature will be called 'MyFeature'
            """
            return 0.0

        @ff.features('MyF1', 'MyF2')
        def MyFeatures(hypothesis):
            """
            this function must return 2 real values 
            in this case 2 features will be computed, they will be named 'MyF1' and 'MyF2', respectively
            note these are also dense features
            """
            return (0.0, 0.0)

        @ff.features('MyF3')
        def MyFeatures(hypothesis):
            """
            this function must return 1 real value
            in this the feature will be named 'MyF3' 
            note how the name of the python function (i.e. MyFeatures) is irrelevant when decorated with ff.features
            """
            return (0.0, 0.0)


