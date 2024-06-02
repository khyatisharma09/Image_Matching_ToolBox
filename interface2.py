from matching import *
import gradio as gr
def ui_change_imagebox(choice):
    """
    Updates the image box with the given choice.

    Args:
        choice (list): The list of image sources to be displayed in the image box.

    Returns:
        dict: A dictionary containing the updated value, sources, and type for the image box.
    """
    return {
        "value": None,  # The updated value of the image box
        "sources": choice,  # The list of image sources to be displayed
        "__type__": "update",  # The type of update for the image box
    }

def ui_reset_state(*args):
    """
    Reset the state of the UI.

    Returns:
        tuple: A tuple containing the initial values for the UI state.
    """
    methods = ["AKAZE", "ASIFT", "BRIEF", "BRISK"," FREAK", "KAZE", "MGM", "ORB", "SGM", "SGBM", "SIFT", "SURF"]
    key = list(methods.keys())[0]  # Get the first key from matcher_zoo
    return (
        None,  # image1
        None,  # image2
        ui_change_imagebox("upload"),  # input image1
        ui_change_imagebox("upload"),  # input image2
        "upload",  # match_image_src
        None,
        None,  # keypoints
        None,  # raw matches
        None,  # ransac matches
        {},  # matches result info
        {},  # matcher config
        None,  # warped image

    )

DESCRIPTION = """
# Image Matching ToolBox
"""
def run():
    with gr.Blocks(css="style.css", theme=gr.themes.Soft()) as gui2:
        gr.Markdown(DESCRIPTION)
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():

                    image1 = gr.Image(label="Query Image",
                                      type="numpy",
                                      height=300,
                                      width=300,
                                      interactive=True
                                      )
                    image2 = gr.Image(label="Train Image",
                                      type="numpy",
                                      height=300,
                                      width=300,
                                      interactive=True
                                      )

                with gr.Row():
                    methods = ["AKAZE","BRISK","KAZE","ORB", "SIFT","SURF"
                               ]
                    method = gr.Dropdown(list(methods),
                                         label="Matching Model",
                                         value="AKAZE")
                    #matcher = gr.Radio(choices=["BF","FLANN"],
                    #                   label="Matcher",
                    #                  value= "BF"
                    #                  )
                with gr.Row():
                    button_reset = gr.Button(value="Reset")
                    button_run = gr.Button(value="Run Match",
                                           variant="primary"
                                           )

                with gr.Accordion("Matching Setting", open=False):
                    with gr.Row():
                        match_setting_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1,
                            step=0.001,
                            label="Match thres.",
                            value=0.1,
                        )
                        match_setting_max_features = gr.Slider(
                            minimum=10,
                            maximum=10000,
                            step=10,
                            label="Max features",
                            value=1000,
                        )
                    with gr.Row():
                        detect_keypoints_threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.001,
                            label="Keypoint thres.",
                            value=0.015,
                        )
                        detect_line_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1,
                            step=0.01,
                            label="Line thres.",
                            value=0.2,
                        )
                    #with gr.Accordion("Geometry Setting", open=False):
                    #    with gr.Row(equal_height=False):
                    #       choice_estimate_geom = gr.Radio(
                    #          ["Fundamental", "Homography"],
                    #          label="Reconstruct Geometry",
                    #         value="Homography",
                    #    )
                    inputs = [image1,
                              image2,
                              method,
                              #match_setting_threshold,
                              #match_setting_max_features,
                              #detect_keypoints_threshold
                              ]
                with gr.Row():
                    gr.Examples(examples=[["demo1.jpg", "demo2.jpg", "AKAZE"],
                                          ["book1.png", "boook2.png", "ORB"],
                                          ["sift1.png", "sift2.png", "SIFT"],
                                        
                                          ],
                                inputs=inputs,
                                outputs=[],
                                label="Examples (click one of the images below to Run"
                            " Match)",
                                fn=matching_function,
                                cache_examples=False,
                                )

            with gr.Column():
                with gr.Row():
                    output1 = gr.Image(label="Keypoints of Query Image",
                                       height=300
                                      )
                    output2 = gr.Image(label="Keypoints of Train Image",
                                        height=300
                                      )

                with gr.Row():
                    output3 = gr.Image(label="Raw Matches")
                with gr.Row():
                    output4 = gr.Image(label="RANSAC Matches")
                with gr.Accordion(
                    "Open for More: Matches Statistics", open=False
                ):
                    matches_result_info = gr.JSON(label="Matches Statistics")

                with gr.Accordion("Open for More: Warped Pair", open=False):
                    with gr.Row():
                        output5 = gr.Image(label="Wrapped Image1")
                        output6 = gr.Image(label="Wrapped Image2")
                with gr.Row():
                    download_button = gr.Button(value="Download", variant="primary")

                outputs = [output1,
                           output2,
                           output3,
                           output4,
                           matches_result_info,
                           output5,
                           output6
                           ]

                button_run.click(
                    fn=matching_function,
                    inputs=inputs,
                    outputs=outputs
                )
                reset_outputs = [
                    image1,
                    image2,
                    match_setting_threshold,
                    match_setting_max_features,
                    detect_keypoints_threshold,
                    image1,
                    image2,
                    output1,
                    output2,
                    output3,
                    output4,
                    matches_result_info,
                    output5,
                    output6
                ]
                button_reset.click(
                    fn=ui_reset_state,
                    inputs=inputs,
                    outputs=reset_outputs
                )
               

    gui2.queue().launch(share=False, debug=True)

run()
