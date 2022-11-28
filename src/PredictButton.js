class PredictButton extends React.Component {
    render() {
        return (
            <button type="button"
                    id={this.props.id}
                    className="btn btn-primary w-50 border border-dark"
                    onClick={this.props.onClick}
                    style={{maxHWidth: '200px',
                        marginTop: '10px',
                        marginBottom: '15px '}}>
                {this.props.isRunning &&
                    <span className="spinner-border spinner-border-sm m-0"></span>
                }
                {this.props.isRunning &&
                    <div className="m-0 p-0">Predicting...</div>
                }
                {!this.props.isRunning &&
                    <div className="m-0 p-0">Predict Points</div>
                }
            </button>
        )
    }
}